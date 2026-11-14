"""
Multilingual YouTube Chatbot for Telegram
-----------------------------------------
1. Create a Telegram bot with @BotFather.
2. Create a .env file beside this script:
       BOT_TOKEN=your_telegram_bot_token
3. Install dependencies:
       python -m pip install -U python-telegram-bot python-dotenv youtube-transcript-api langchain langchain-community langchain-text-splitters langchain-huggingface sentence-transformers faiss-cpu langchain-ollama
4. Install Ollama and download the model:
       ollama pull llama3.2
5. Run:
       python bot.py

Telegram commands:
    /start
    /load https://www.youtube.com/watch?v=VIDEO_ID
    /clear
    /help

The bot prefers captions in Hindi, English, Bhojpuri, Bengali, Telugu,
Marathi, Tamil, Gujarati, Kannada, Malayalam, and Punjabi.
If none of these is available, it tries the first available caption language.
"""

import asyncio
import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


# -------------------- Configuration --------------------

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVER_K = 4
TELEGRAM_MESSAGE_LIMIT = 4000

# Language preference order. The first available language in this list wins.
PREFERRED_LANGUAGES = [
    "hi", "en", "bho", "bn", "te", "mr",
    "ta", "gu", "kn", "ml", "pa",
]

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

if not BOT_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN is missing. Create a .env file beside bot.py containing:\n"
        "BOT_TOKEN=your_telegram_bot_token"
    )

# These models are loaded once when the bot starts.
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)


# -------------------- YouTube / RAG functions --------------------

VIDEO_ID_PATTERN = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|shorts/|live/))"
    r"([A-Za-z0-9_-]{11})"
)


def extract_video_id(value: str) -> Optional[str]:
    """Extract an 11-character YouTube video ID from a URL or accept an ID."""
    value = value.strip()

    # Accept a video ID by itself.
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    match = VIDEO_ID_PATTERN.search(value)
    return match.group(1) if match else None


def load_video_into_faiss(video_url: str) -> tuple[FAISS, str, str]:
    """
    Fetch the preferred available transcript (or another available language)
    and create a FAISS vector store.
    """
    video_id = extract_video_id(video_url)

    if not video_id:
        raise ValueError(
            "That doesn't look like a valid YouTube URL. Please send a link such as "
            "https://www.youtube.com/watch?v=VIDEO_ID"
        )

    # Get the transcripts/captions available for this video.
    transcript_list = YouTubeTranscriptApi().list(video_id)
    available_transcripts = list(transcript_list)

    if not available_transcripts:
        raise ValueError("No captions are available for this video.")

    selected_transcript = None

    # Prefer the requested Indian languages and English.
    for language in PREFERRED_LANGUAGES:
        selected_transcript = next(
            (
                transcript
                for transcript in available_transcripts
                if transcript.language_code.lower() == language
            ),
            None,
        )
        if selected_transcript is not None:
            break

    # If no preferred language is available, use the first available caption.
    if selected_transcript is None:
        selected_transcript = available_transcripts[0]

    transcript_items = selected_transcript.fetch()

    transcript_text = " ".join(
        item.text
        for item in transcript_items
        if getattr(item, "text", "").strip()
    )

    if not transcript_text.strip():
        raise ValueError("The selected transcript is empty.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    documents = splitter.create_documents([transcript_text])

    if not documents:
        raise ValueError("I couldn't create searchable chunks from this transcript.")

    vector_store = FAISS.from_documents(documents, embeddings)
    language_name = selected_transcript.language
    language_code = selected_transcript.language_code

    return vector_store, video_id, f"{language_name} ({language_code})"


def answer_from_video(vector_store: FAISS, question: str) -> str:
    """Retrieve relevant transcript chunks and ask Ollama for a grounded answer."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )
    retrieved_docs: list[Document] = retriever.invoke(question)

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""
You are a helpful multilingual YouTube video assistant.

You can understand and answer in Hindi, English, Bhojpuri, Bengali,
Telugu, Marathi, Tamil, Gujarati, Kannada, Malayalam, Punjabi, and Hinglish.

Instructions:
- Detect the language of the user's question.
- Reply in the same language as the user's question.
- If the user asks in Hinglish, reply in natural Hinglish.
- Use only the transcript context below to answer questions about the video.
- Do not invent facts or claim that something is in the video if it is not in the context.
- If the answer is not available in the provided context, say so clearly.
- You may explain the transcript in simple language when asked.

Transcript context:
{context_text}

User question:
{question}

Answer:
"""

    result = llm.invoke(prompt)
    return str(result.content).strip()


# -------------------- Telegram handlers --------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message:
        await message.reply_text(
            "👋 Welcome to the Multilingual YouTube Chatbot!\n\n"
            "1. Send /load followed by a YouTube video URL.\n"
            "2. I'll find an available caption transcript.\n"
            "3. Ask questions in Hindi, English, Bhojpuri, or other supported languages.\n\n"
            "Example:\n"
            "/load https://www.youtube.com/watch?v=VIDEO_ID\n\n"
            "Use /help to see commands."
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message:
        await message.reply_text(
            "📚 Commands\n\n"
            "/start - Start the bot\n"
            "/load VIDEO_URL - Load a YouTube video\n"
            "/clear - Clear the loaded video\n"
            "/help - Show this help\n\n"
            "After loading a video, send your question as a normal message.\n"
            "The bot prefers Hindi, English, Bhojpuri, Bengali, Telugu, Marathi, "
            "Tamil, Gujarati, Kannada, Malayalam, and Punjabi captions. "
            "If none is available, it tries another available caption language."
        )


async def load_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return

    if not context.args:
        await message.reply_text(
            "Please provide a YouTube URL.\n"
            "Example: /load https://www.youtube.com/watch?v=VIDEO_ID"
        )
        return

    video_url = " ".join(context.args).strip()
    await message.reply_text(
        "⏳ Checking available captions and building the search index..."
    )

    try:
        vector_store, video_id, transcript_language = await asyncio.to_thread(
            load_video_into_faiss,
            video_url,
        )

        context.user_data["vector_store"] = vector_store
        context.user_data["video_id"] = video_id
        context.user_data["transcript_language"] = transcript_language

        await message.reply_text(
            "✅ Video loaded successfully!\n\n"
            f"Caption language: {transcript_language}\n\n"
            "You can now ask questions about this video's transcript."
        )

    except TranscriptsDisabled:
        await message.reply_text(
            "This video has captions/transcripts disabled, so I can't read its content."
        )
    except NoTranscriptFound:
        await message.reply_text(
            "I couldn't find any accessible captions for this video. "
            "Please try another video."
        )
    except VideoUnavailable:
        await message.reply_text(
            "This video is unavailable or cannot be accessed."
        )
    except ValueError as exc:
        await message.reply_text(f"⚠️ {exc}")
    except Exception as exc:
        logger.exception("Could not load YouTube video")
        await message.reply_text(
            "Sorry, I couldn't load this video. Check that it has accessible captions "
            "and try again.\n\n"
            f"Technical detail: {str(exc)[:500]}"
        )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    context.user_data.pop("vector_store", None)
    context.user_data.pop("video_id", None)
    context.user_data.pop("transcript_language", None)

    if message:
        await message.reply_text("🧹 The loaded video has been cleared.")


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message or not message.text:
        return

    vector_store = context.user_data.get("vector_store")
    if vector_store is None:
        await message.reply_text(
            "First load a YouTube video using:\n"
            "/load https://www.youtube.com/watch?v=VIDEO_ID"
        )
        return

    question = message.text.strip()
    if not question:
        return

    await context.bot.send_chat_action(
        chat_id=message.chat_id,
        action=ChatAction.TYPING,
    )

    try:
        answer = await asyncio.to_thread(
            answer_from_video,
            vector_store,
            question,
        )

        # Split long answers because Telegram limits message length.
        for start_index in range(0, len(answer), TELEGRAM_MESSAGE_LIMIT):
            await message.reply_text(
                answer[start_index : start_index + TELEGRAM_MESSAGE_LIMIT]
            )

    except Exception:
        logger.exception("Error while answering a question")
        await message.reply_text(
            "Sorry, I couldn't generate an answer. Make sure Ollama is running "
            "and the llama3.2 model is installed."
        )


# -------------------- Start bot --------------------

def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("load", load_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, chat)
    )

    logger.info("Multilingual YouTube chatbot is starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
