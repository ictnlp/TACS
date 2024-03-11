from gradio_highlightedtextbox import HighlightedTextbox
from fastchat.conversation import get_conv_template

SYS_MSG = """According to the given information and your knowledge, answer the question."""
QUESTION_TEMPLATE = "Question: {}"
INFORMATION_TEMPLATE = "Information: {}"
ANSWER = "\nAnswer:"


def convert_tagged_text_to_highlighted_text(
    tagged_text=""
):
    return HighlightedTextbox.tagged_text_to_tuples(
        tagged_text, ['Truthful', 'Untruthful'], ['<a>', '<b>'], ['</a>', '</b>']
    )

def format_prompt(question, information):
    prompt_template = get_conv_template('llama-2')
    prompt_template.set_system_message(SYS_MSG)
    prompt_template.append_message(prompt_template.roles[0], QUESTION_TEMPLATE.format(question)+"\n"+INFORMATION_TEMPLATE.format(information)+"\n"+ANSWER)
    prompt_template.append_message(prompt_template.roles[1], "")
    return prompt_template.get_prompt()