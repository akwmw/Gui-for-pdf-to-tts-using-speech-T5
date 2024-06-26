import unittest
from unittest import mock
import PyPDF2
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

from project import process_pdf

class TestProcessPdf(unittest.TestCase):
    def setUp(self):
        self.file_path = 'test.pdf'
        self.text = "This is a sample text from the PDF document."

        # Set up mocks
        with open(self.file_path, 'w') as f:
            f.write(self.text)

        self.pypdf2_mock = mock.Mock(spec=PyPDF2.PdfFileReader)
        self.processor_mock = mock.Mock(spec=SpeechT5Processor)
        self.model_mock = mock.Mock(spec=SpeechT5ForTextToSpeech)
        self.vocoder_mock = mock.Mock(spec=SpeechT5HifiGan)
        self.embeddings_dataset_mock = mock.Mock()

    def test_process_pdf(self):
        # Given
        selected_tool = "pypdf2"
        with open(self.file_path, 'rb') as file:
            self.pypdf2_mock.getPage.return_value.extractText.return_value = self.text

        # When
        process_pdf(self.file_path, self.pypdf2_mock, self.processor_mock, self.model_mock, self.vocoder_mock,
                    self.embeddings_dataset_mock)

        # Then
        self.pypdf2_mock.getPage.assert_called_once_with(0)
        self.assertEqual(self.pypdf2_mock.getPage.return_value.extractText.call_count, 1)
        """
    Fail condion:
    If process_pdf does not call the getPage method of the PyPDF2.PdfFileReader object.
    If process_pdf calls the getPage method more than once or with an argument other than 0.
    If process_pdf does not call the extractText method of the page object.
    If process_pdf calls the extractText method more than once.
    """

def test_remove_before_abstract():
    text = "Introduction... Abstract This is the abstract."
    assert remove_before_abstract(text) == "Abstract This is the abstract."

def test_remove_after_ack_or_ref():
    text = "This is the main body. Acknowledgements Thanks to everyone."
    assert remove_after_ack_or_ref(text) == "This is the main body. "

def test_remove_table():
    text = "Some text. \\begin{table} This is a table \\end{table} More text."
    assert remove_table(text) == "Some text.  More text."

def test_remove_references():
    text = "As stated in [1; 2]."
    assert remove_references(text) == "As stated in ."




