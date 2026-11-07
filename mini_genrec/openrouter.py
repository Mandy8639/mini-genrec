"""List currently advertised named free models: python -m mini_genrec.openrouter."""
import json
from urllib.request import Request, urlopen


def free_models():
    request = Request("https://openrouter.ai/api/v1/models", headers={"User-Agent": "mini-genrec"})
    with urlopen(request, timeout=30) as response:
        models = json.load(response)["data"]
    return sorted((model for model in models if model["id"].endswith(":free")),
                  key=lambda model: model["id"])


def main():
    print("Model ID\tAdvertised structured-output support")
    for model in free_models():
        parameters = model.get("supported_parameters", [])
        support = ("structured_outputs advertised; smoke-test endpoint" if "structured_outputs" in parameters
                   else "response_format only; JSON schema may be unavailable" if "response_format" in parameters
                   else "not advertised")
        print(f"{model['id']}\t{support}")


if __name__ == "__main__":
    main()
