from src.ingestion.document_parser import TextProcessor
from src.config import settings
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__, level=settings.LOG_LEVEL.upper() if hasattr(settings, 'LOG_LEVEL') else 'INFO')

async def main_test_doc():
    # Create a dummy PDF and a text file for demonstration
    from fpdf import FPDF

    # Create a dummy PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(
        0,
        10,
        "This is the first page of the document. It contains some sample text. "
        "The second page will have more text. This is a test for the TextProcessor class. "
        "We will extract this text and process it.",
    )
    pdf.add_page()
    pdf.multi_cell(
        0,
        10,
        "This is the second page. It continues the sample text with more information. "
        "We will test text cleaning and chunking on this combined text. "
        "The quick brown fox jumps over the lazy dog.",
    )
    pdf.output("./data/files/sample.pdf")

    # Create a dummy text file
    with open("./data/files/sample.txt", "w") as f:
        f.write(
            "I don't know. We should decide which kind of remote control we want to go. Should it be specific remote control to some specific device? Should it be a universal one? And etc. So, I'm waiting for your inputs very quickly because we have only three minutes to go. Okay. Well. The first thing that they've kind of specified is the price like based on how much profit we want to make, which seems to be kind of a little strange if we don't know what the product is yet, but I guess if that's the requirement that we need to design the product to actually fit that price bracket. So, I guess we're going to need to find out what's actually, you know, what people are willing to pay for, what kind of product they're expecting for 25 euro. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. I think this is more a job to our marketing person. So, it should be the topic of maybe also the next meeting just to have an overview of this and in which direction we should go. So, we need to close the meeting. We'll have a new meeting soon and so all the work every of you have to do. So, you have to work on the working design. You have to... Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. I'm going to work on the technical functions and you have to work on user requirements specs. Alright. Yeah. You will receive some information by email as usual. Thanks for coming today. Okay, thanks. Thanks. Alright. I'm going to go get a pen."
        )

    # Initialize the TextProcessor
    processor = TextProcessor()

    # 1. PDF Text Extraction
    pdf_text = processor.extract_text_from_pdf("./data/files/sample.pdf")
    logger.info(pdf_text)

    # 2. Plain Text File Reading
    text_file_content = processor.read_text_file("./data/files/sample.txt")
    logger.info(text_file_content)

    # 3. Text Cleaning
    cleaned_text = processor.clean_text(text_file_content, stem_method="spacy")
    logger.info(cleaned_text)

    # 4. Text Chunking
    text_chunks = processor.chunk_text(pdf_text)
    for i, chunk in enumerate(text_chunks):
        logger.info(f"Chunk {i+1}: {chunk}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main_test_doc())