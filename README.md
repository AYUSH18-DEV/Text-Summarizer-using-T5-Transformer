# Text Summarization using T5 Transformer

A Transformer-based text summarization application built using the T5 (Text-to-Text Transfer Transformer) model. The application takes lengthy text as input and generates a concise and meaningful summary.

The trained T5 model is integrated with a FastAPI backend to provide a simple web-based interface for generating summaries.

## Features

* Text summarization using the T5 Transformer model
* Natural Language Processing using Hugging Face Transformers
* Model inference using PyTorch
* Text preprocessing and cleaning
* FastAPI backend for serving the trained model
* Web interface for entering text and generating summaries
* Local trained model support

## About T5

T5 (Text-to-Text Transfer Transformer) is a Transformer-based model developed by Google for solving different Natural Language Processing tasks using a text-to-text approach.

For summarization, the input is provided to the model with a task prefix:

```text
summarize: <input text>
```

The model processes the input and generates a concise summary as the output.

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* T5 Transformer
* FastAPI
* Jinja2
* HTML
* CSS
* Uvicorn

## Project Workflow

1. User enters the text through the web interface.
2. The text is sent to the FastAPI backend.
3. The input text is cleaned and preprocessed.
4. The T5 tokenizer converts the text into tokens.
5. The tokens are passed to the trained T5 model.
6. The model generates the summary.
7. The generated tokens are decoded into readable text.
8. The summary is displayed to the user.

## Running the Application

Start the FastAPI application using Uvicorn:

```bash
uvicorn app:app --reload
```

After starting the server, open the following address in your browser:

```text
http://127.0.0.1:8000
```

## Example

### Input

```text
Artificial intelligence is transforming various industries by automating repetitive tasks, improving decision making, and providing intelligent solutions to complex problems.
```

### Output

```text
Artificial intelligence is transforming industries through automation and intelligent decision making.
```

## Project Objective

The objective of this project is to develop an end-to-end Natural Language Processing application that can automatically generate concise summaries from lengthy text.

The project demonstrates the complete workflow from text preprocessing and Transformer-based model inference to deploying the model through a FastAPI web application.

## Key Concepts

* Natural Language Processing
* Text preprocessing
* Tokenization
* Transformer architecture
* T5 Encoder-Decoder architecture
* Sequence-to-sequence learning
* Text generation
* Model inference
* FastAPI API development
* Web application integration

## Future Improvements

* Deploy the application on a cloud platform
* Add support for multiple languages
* Improve summarization quality through additional fine-tuning
* Add summary length control
* Add PDF and document summarization
* Optimize model inference speed
* Add API authentication and rate limiting

## Author

Ayush Singh
