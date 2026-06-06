from autogen_agentchat.agents import AssistantAgent


def create_writer(model_client):

    return AssistantAgent(
        name="writer",
        model_client=model_client,
        system_message=(
            "Write a poem."
        )
    )


def create_reviewer(model_client):

    return AssistantAgent(
        name="reviewer",
        model_client=model_client,
        system_message=(
            "Review the poem."
        )
    )