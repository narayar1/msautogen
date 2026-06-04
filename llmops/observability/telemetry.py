from opentelemetry import trace

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from config import SERVICE_NAME


def initialize_tracer():

    print("Open Telemetry: Initializing tracer")
    resource = Resource.create(
        {
            "service.name": SERVICE_NAME
        }
    )

    provider = TracerProvider(
        resource=resource
    )

    trace.set_tracer_provider(provider)

    return trace.get_tracer(
        SERVICE_NAME
    )