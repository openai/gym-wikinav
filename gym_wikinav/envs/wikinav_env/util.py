import math

import requests
from tqdm import trange


def download_file(url, destination=None, chunk_size=1024):
    if destination is None:
        destination = url.split("/")[-1]
    r = requests.get(url, stream=True)
    with open(destination, "wb") as f:
        size = int(r.headers["content-length"])
        n_chunks = math.ceil(size / float(chunk_size))
        r_iter = r.iter_content(chunk_size=chunk_size)

        for _ in trange(n_chunks):
            chunk = next(r_iter)
            if chunk:
                f.write(chunk)

        # HACK: keep going in case we somehow missed chunks; maybe a wrong
        # header or the like
        for chunk in r_iter:
            if chunk:
                f.write(chunk)

    return destination
