import json
import chainlit as cl
from graph_builder import build_graph
from vector_db import check_collection

@cl.on_chat_start
async def on_chat_start():
    check_collection()
    cl.user_session.set("app", build_graph())
    await cl.Message("System aktywny. Zadaj pytanie dotyczące leków.").send()

@cl.on_message
async def on_message(message: cl.Message):
    app = cl.user_session.get("app")
    if app is None:
        await cl.Message(content="Błąd. Zrestartuj system.").send()
        return

    inputs = {"question": message.content}
    final_output = app.invoke(inputs)  
    generation = final_output.get("generation")

    if generation:
        try:
            parsed = json.loads(generation)
            generation = parsed.get("answer", generation)
        except (json.JSONDecodeError, TypeError):
            pass

        await cl.Message(content=generation).send()
    else:
        await cl.Message(content="Przepraszam nie mogłem wygenerować odpowiedzi.").send()
