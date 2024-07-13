from datachain.lib.feature import Feature


class Usage(Feature):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class MyChatMessage(Feature):
    role: str = ""
    content: str = ""


class CompletionResponseChoice(Feature):
    message: MyChatMessage = MyChatMessage()


class MistralModel(Feature):
    id: str = ""
    choices: list[CompletionResponseChoice]
    usage: Usage = Usage()
