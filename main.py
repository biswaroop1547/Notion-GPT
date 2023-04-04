"""Main entrypoint for the app."""
import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from langchain.chains import ChatVectorDBChain
from query_data import get_chain
from schemas import ChatResponse

from langchain.schema import Document
from typing import List

import concurrent

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None


def format_references(references):
    references = [f"\r- {title}: {url}" for title, url in references]
    return references

def get_references(documents: List[Document]) -> List[Document]:
    references = []
    for doc in documents:
        *title, uuid = doc.metadata["source"].split("/")[-1].split(" ")
        uuid = uuid.replace(".md", "")
        url = "https://www.notion.so/skit-ai/" + uuid
        references.append((" ".join(title), url))
    return format_references(references)


@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    if not Path("vectorstore.pkl").exists():
        raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
    with open("vectorstore.pkl", "rb") as f:
        global vectorstore
        vectorstore = pickle.load(f)


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain: ChatVectorDBChain = get_chain(vectorstore, question_handler, stream_handler)
    executor = concurrent.futures.ThreadPoolExecutor()
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())
            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history},
                executor=executor
            )
            chat_history.append((question, result["answer"]))
            
            first_two_reference_docs: List[Document] = result["source_documents"][:2]
            refs = " Found References: \n" + "\n".join(get_references(documents=first_two_reference_docs))
            references_resp = ChatResponse(sender="bot", message=refs, type="stream")
            await websocket.send_json(references_resp.dict())
            
            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
