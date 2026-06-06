from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider


def initialize_tracer(service_name: str):
    """
    Initialize and return an OpenTelemetry tracer.

    Args:
        service_name: Name of the application/service.

    Returns:
        OpenTelemetry tracer instance.
    """

    print(f"OpenTelemetry: Initializing tracer for {service_name}")

    resource = Resource.create(
        {
            "service.name": service_name
        }
    )

    provider = TracerProvider(resource=resource)

    try:
        trace.set_tracer_provider(provider)
    except Exception:
        # Provider may already be initialized
        pass

    return trace.get_tracer(service_name)


def get_trace_context():
    """
    Get current trace and span IDs from active span.

    Returns:
        dict containing trace_id and span_id
    """

    current_span = trace.get_current_span()
    ctx = current_span.get_span_context()

    if not ctx.is_valid:
        return {
            "trace_id": None,
            "span_id": None
        }

    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x")
    }