"""Bounded health/model/inference verification without third-party HTTP clients."""
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(url, payload=None, timeout=10):
    request = Request(url, data=None if payload is None else json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
        return json.loads(body) if body else {}


def verify_endpoint(endpoint, *, process=None, timeout=900, deployed=False, on_status=None):
    """A URL alone is insufficient; require actual model discovery and completion."""
    deadline = time.monotonic() + timeout
    last_error = 'endpoint has not responded'
    while time.monotonic() < deadline:
        if process is not None:
            code = process.poll()
            if code is not None and (not deployed or code != 0):
                raise RuntimeError(f'Serving process exited before readiness (exit {code})')
        try:
            remaining = max(0.1, min(30, deadline - time.monotonic()))
            request_json(endpoint.rstrip('/') + '/health', timeout=remaining)
            if on_status:
                on_status('Health OK; checking served models')
            models = request_json(endpoint.rstrip('/') + '/v1/models',
                                  timeout=max(0.1, min(30, deadline-time.monotonic())))
            model = models['data'][0]['id']
            if on_status:
                on_status('Models OK; verifying actual inference')
            result = request_json(endpoint.rstrip('/') + '/v1/completions',
                                  dict(model=model, prompt='Hello', max_tokens=1, temperature=0),
                                  timeout=max(0.1, min(60, deadline - time.monotonic())))
            if not isinstance(result.get('choices'), list) or not result['choices']:
                raise ValueError('Completion response has no choices')
            return model
        except (OSError, URLError, HTTPError, ValueError, KeyError, IndexError) as exc:
            last_error = str(exc)
            if on_status:
                on_status('Waiting for endpoint: ' + last_error[:160])
            time.sleep(min(2, max(0, deadline-time.monotonic())))
    raise RuntimeError(f'Endpoint readiness timed out: {last_error}')
