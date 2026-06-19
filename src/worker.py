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
                        "metadata": {
                            "title": title,
                            "link": link,
                            "description": description[:500],
                        },
                    }],
                    dict_converter=Object.fromEntries,
                )

                await self.env.VECTOR_DB.upsert(vector_payload)
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

    async def _query(self, request: Request) -> Response:
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
        except Exception:
            return Response(
                json.dumps({"error": "Invalid JSON body"}),
                status=400,
                headers={"Content-Type": "application/json"},
            )

        if not query_text:
            return Response(
                json.dumps({"error": "Missing 'query' field"}),
                status=400,
                headers={"Content-Type": "application/json"},
            )

        api_key = getattr(self.env, "MISTRAL_API_KEY")

        try:
            embedding = await self._embed(api_key, query_text)
        except Exception as e:
            return Response(
                json.dumps({"error": f"Embedding failed: {str(e)}"}),
                status=502,
                headers={"Content-Type": "application/json"},
            )

        try:
            query_vector = to_js(embedding)
            results = await self.env.VECTOR_DB.query(
                query_vector,
                to_js({"topK": top_k, "returnMetadata": "all"}, dict_converter=Object.fromEntries),
            )
            matches = [
                {
                    "id": m.id,
                    "score": m.score,
                    "metadata": m.metadata.to_py() if hasattr(m.metadata, "to_py") else {},
                }
                for m in results.matches
            ]
        except Exception as e:
            return Response(
                json.dumps({"error": f"Query failed: {str(e)}"}),
                status=500,
                headers={"Content-Type": "application/json"},
            )

        return Response(
            json.dumps({"results": matches}),
            headers={"Content-Type": "application/json"},
        )

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
