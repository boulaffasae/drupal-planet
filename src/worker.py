from pyodide.ffi import to_js
from js import Object
import json
from workers import WorkerEntrypoint, Request, Response, fetch
from bs4 import BeautifulSoup
import hashlib


class Default(WorkerEntrypoint):
    async def fetch(self, request: Request) -> Response:
        path = request.url.split("?")[0].rstrip("/").rsplit("/", 1)[-1]

        if path == "ingest":
            return await self._ingest(request)
        if path == "query":
            return await self._query(request)

        return Response(
            json.dumps({"error": "Not found"}),
            status=404,
            headers={"Content-Type": "application/json"},
        )

    async def _ingest(self, request: Request) -> Response:
        if request.method != "POST":
            return Response(
                json.dumps({"error": "Method not allowed"}),
                status=405,
                headers={"Content-Type": "application/json"},
            )

        cron_secret = getattr(self.env, "CRON_SECRET", None)
        auth_header = request.headers.get("Authorization")
        if not cron_secret or auth_header != cron_secret:
            return Response(
                json.dumps({"error": "Unauthorized"}),
                status=401,
                headers={"Content-Type": "application/json"},
            )

        try:
            body = await request.json()
            url = body.get("url")
        except Exception:
            return Response(
                json.dumps({"error": "Invalid JSON body"}),
                status=400,
                headers={"Content-Type": "application/json"},
            )

        if not url:
            return Response(
                json.dumps({"error": "Missing 'url' field"}),
                status=400,
                headers={"Content-Type": "application/json"},
            )

        try:
            feed_response = await fetch(url)
            content = await feed_response.text()
        except Exception as e:
            return Response(
                json.dumps({"error": f"Failed to fetch feed: {str(e)}"}),
                status=502,
                headers={"Content-Type": "application/json"},
            )

        soup = BeautifulSoup(content, "xml")
        items = soup.find_all("item")

        if not items:
            return Response(
                json.dumps({"error": "No items found in feed"}),
                status=422,
                headers={"Content-Type": "application/json"},
            )

        api_key = getattr(self.env, "MISTRAL_API_KEY")
        processed = 0
        errors = []

        for item in items:
            title_tag = item.find("title")
            link_tag = item.find("link")
            description_tag = item.find("description")

            title = title_tag.get_text(strip=True) if title_tag else ""
            link = link_tag.get_text(strip=True) if link_tag else ""
            description = description_tag.get_text(strip=True) if description_tag else ""

            if not link:
                errors.append({"item": title, "error": "missing link, skipped"})
                continue

            already_processed = await self.env.D_ONE.prepare(
                "SELECT 1 FROM nodes WHERE link = ?"
            ).bind(str(link)).first()
            if already_processed:
                continue

            text = f"{title}\n{description}".strip()
            if not text:
                errors.append({"item": link, "error": "empty content, skipped"})
                continue

            vector_id = hashlib.md5(link.encode()).hexdigest()

            try:
                embedding = await self._embed(api_key, text)

                if len(embedding) != 1024:
                    errors.append({"item": link, "error": f"bad embedding dims ({len(embedding)})"})
                    continue

                vector_payload = to_js(
                    [{
                        "id": vector_id,
                        "values": embedding,
                    }],
                    dict_converter=Object.fromEntries,
                )

                await self.env.VECTOR_DB.upsert(vector_payload)

                await self.env.D_ONE.prepare(
                    """
                    INSERT OR REPLACE INTO nodes (id, link, title, description) 
                    VALUES (?, ?, ?, ?)
                    """
                ).bind(str(vector_id), str(link), str(title), str(description)).run()

                processed += 1
            except Exception as e:
                errors.append({"item": link, "error": str(e)})
                continue

        return Response(
            json.dumps({
                "status": "success",
                "processed": processed,
                "total": len(items),
                "errors": errors,
            }),
            headers={"Content-Type": "application/json"},
        )

    _CORS_HEADERS = {
        "Access-Control-Allow-Origin": "https://abzhcompany.com",
        "Access-Control-Allow-Methods": "POST",
        "Access-Control-Allow-Headers": "Accept, Content-Type, Authorization",
    }

    async def _query(self, request: Request) -> Response:
        if request.method == "OPTIONS":
            return Response("", status=204, headers=self._CORS_HEADERS)

        if request.method != "POST":
            return Response(
                json.dumps({"error": "Method not allowed"}),
                status=405,
                headers={"Content-Type": "application/json"},
            )

        try:
            body = await request.json()
            query_text = body.get("query")
            top_k = int(body.get("top_k", 5))
            turnstile_token = body.get("token")
        except Exception:
            return Response(
                json.dumps({"error": "Invalid JSON body"}),
                status=400,
                headers={"Content-Type": "application/json", **self._CORS_HEADERS},
            )

        if not query_text:
            return Response(
                json.dumps({"error": "Missing 'query' field"}),
                status=400,
                headers={"Content-Type": "application/json", **self._CORS_HEADERS},
            )

        if not turnstile_token:
            return Response(
                json.dumps({"error": "Missing Turnstile token"}),
                status=400,
                headers={"Content-Type": "application/json", **self._CORS_HEADERS},
            )

        remote_ip = request.headers.get("CF-Connecting-IP")
        turnstile_ok = await self._verify_turnstile(turnstile_token, remote_ip)
        if not turnstile_ok:
            return Response(
                json.dumps({"error": "Turnstile verification failed"}),
                status=403,
                headers={"Content-Type": "application/json", **self._CORS_HEADERS},
            )

        api_key = getattr(self.env, "MISTRAL_API_KEY")

        try:
            embedding = await self._embed(api_key, query_text)
        except Exception as e:
            return Response(
                json.dumps({"error": f"Embedding failed: {str(e)}"}),
                status=502,
                headers={"Content-Type": "application/json", **self._CORS_HEADERS},
            )

        try:
            query_vector = to_js(embedding)
            results = await self.env.VECTOR_DB.query(
                query_vector,
                to_js({"topK": top_k, "returnMetadata": "all"}, dict_converter=Object.fromEntries),
            )

            raw_matches = list(results.matches)
            # It's better to keep track of scores alongside IDs
            scores = {str(m.id): m.score for m in raw_matches}
            match_ids = list(scores.keys())

            if not match_ids:
                return Response(
                    json.dumps({"results": []}),
                    headers={"Content-Type": "application/json", **self._CORS_HEADERS},
                )

            # 1. Fetch from D1
            placeholders = ", ".join(["?"] * len(match_ids))
            d1_results = await self.env.D_ONE.prepare(
                f"SELECT id, title, link, description FROM nodes WHERE id IN ({placeholders})"
            ).bind(*match_ids).all()
            
            # 2. Safely convert to a standard Python list of dicts
            db_rows = d1_results.results.to_py()

            # 3. Ensure IDs are strings for reliable mapping
            rows_by_id = {str(row["id"]): row for row in db_rows}
            
            # 4. Build final matches and include the vector score
            matches = []
            for vid in match_ids:
                if vid in rows_by_id:
                    row = rows_by_id[vid]
                    matches.append({
                        "id": vid,
                        "title": row.get("title"),
                        "link": row.get("link"),
                        "description": row.get("description"),
                        "score": scores[vid]  # Highly recommended to pass this back to the client!
                    })
        except Exception as e:
            return Response(
                json.dumps({"error": f"Query failed: {str(e)}"}),
                status=500,
                headers={"Content-Type": "application/json", **self._CORS_HEADERS},
            )

        return Response(
            json.dumps({"results": matches}),
            headers={"Content-Type": "application/json", **self._CORS_HEADERS},
        )

    async def _verify_turnstile(self, token: str, remote_ip: str | None) -> bool:
        secret = getattr(self.env, "TURNSTILE_SECRET_KEY")
        payload = {"secret": secret, "response": token}
        if remote_ip:
            payload["remoteip"] = remote_ip

        response = await fetch(
            "https://challenges.cloudflare.com/turnstile/v0/siteverify",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=json.dumps(payload),
        )
        data = await response.json()
        return bool(data.get("success"))

    async def _embed(self, api_key: str, text: str) -> list:
        response = await fetch(
            "https://api.mistral.ai/v1/embeddings",
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            body=json.dumps({
                "model": "mistral-embed",
                "input": [text],
            }),
        )

        if not response.ok:
            body_text = await response.text()
            raise Exception(f"Mistral API error {response.status}: {body_text}")

        data = await response.json()
        return list(data["data"][0]["embedding"])
