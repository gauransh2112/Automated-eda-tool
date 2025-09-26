# client.py
import requests, base64, os, sys

SERVER = "http://127.0.0.1:8000"

def post_prompt(file_path, prompt):
    resp = requests.post(f"{SERVER}/eda", json={"file_path": file_path, "prompt": prompt})
    return resp.json()

def save_plot(data_uri, fname):
    header, b64 = data_uri.split(",", 1) if "," in data_uri else (None, data_uri)
    img = base64.b64decode(b64)
    with open(fname, "wb") as f:
        f.write(img)
    print("Saved", fname)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python client.py examples/iris.csv \"distribution of sepal_length\"")
        sys.exit(1)
    file_path = sys.argv[1]
    prompt = sys.argv[2]
    out = post_prompt(file_path, prompt)
    print(out.get("result") or out.get("error"))
    if out.get("status") == "ok":
        res = out["result"]
        if isinstance(res, dict) and "plot" in res:
            save_plot(res["plot"], "out.png")
