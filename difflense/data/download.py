"""Downloads without a browser (Colab, clusters). Resumable; files already complete are skipped.

qg   : CERNBox public share (ML4Sci CMS quark/gluon jet images, 3 parquet files, ~3.7 GB), listed over the
       public WebDAV endpoint and fetched file by file.
calo : CaloChallenge 2022 Dataset 2 from Zenodo record 6366271, md5-checked.
"""
import hashlib
import os
import shutil
import subprocess
import urllib.parse
import xml.etree.ElementTree as ET

import requests

CERNBOX = "https://cernbox.cern.ch"
DATA_EXTS = (".parquet", ".h5", ".hdf5", ".npz", ".pt")
_PROPFIND = ('<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop>'
             '<d:resourcetype/><d:getcontentlength/></d:prop></d:propfind>')


def fetch(url, dest, size=None):
    if os.path.exists(dest) and size and os.path.getsize(dest) == size:
        print(f"[skip] {os.path.basename(dest)} already complete")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[get ] {url}\n    -> {dest}")
    if shutil.which("wget"):
        subprocess.run(["wget", "-c", "-q", "--show-progress", "--progress=bar:force:noscroll", "-O", dest, url],
                       check=True)
    else:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
    if size and os.path.getsize(dest) != size:
        raise RuntimeError(f"{dest}: {os.path.getsize(dest)} bytes, expected {size}; re-run to resume")
    return dest


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def list_cernbox(token, sub=""):
    """Recursively list a public CERNBox share -> [(relative path, size in bytes)]."""
    base = f"{CERNBOX}/remote.php/dav/public-files/{token}"
    url = f"{base}/{urllib.parse.quote(sub)}" if sub else base + "/"
    r = requests.request("PROPFIND", url, data=_PROPFIND, headers={"Depth": "1"}, timeout=60)
    r.raise_for_status()
    ns = {"d": "DAV:"}
    out = []
    for resp in ET.fromstring(r.content).findall("d:response", ns):
        rel = urllib.parse.unquote(resp.find("d:href", ns).text).split(token, 1)[-1].strip("/")
        if rel == sub.strip("/"):
            continue
        if resp.find(".//d:resourcetype/d:collection", ns) is not None:
            out += list_cernbox(token, rel)
        else:
            s = resp.find(".//d:getcontentlength", ns)
            out.append((rel, int(s.text) if s is not None and s.text else None))
    return out


def download_qg(cfg):
    token = cfg["data"]["cernbox_token"]
    out = cfg["paths"]["raw"]
    listing = [(r, s) for r, s in list_cernbox(token) if r.lower().endswith(DATA_EXTS)]
    print(f"CERNBox share: {len(listing)} data files, {sum(s or 0 for _, s in listing) / 1e9:.2f} GB")
    for rel, size in listing:
        url = f"{CERNBOX}/remote.php/dav/public-files/{token}/{urllib.parse.quote(rel)}"
        fetch(url, os.path.join(out, rel), size)
    return [os.path.join(out, r) for r, _ in listing]


def download_calo(cfg):
    d = cfg["data"]
    record = d.get("zenodo_record", "6366271")
    meta = {}
    try:
        r = requests.get(f"https://zenodo.org/api/records/{record}", timeout=30)
        r.raise_for_status()
        meta = {f["key"]: f for f in r.json()["files"]}
    except Exception as e:
        print(f"Zenodo API unavailable ({e}); downloading without size/md5 checks")
    paths = []
    for name in d["files"]:
        info = meta.get(name, {})
        dest = fetch(f"https://zenodo.org/records/{record}/files/{name}?download=1",
                     os.path.join(cfg["paths"]["raw"], name), info.get("size"))
        if info.get("checksum", "").startswith("md5:"):
            if md5sum(dest) != info["checksum"][4:]:
                raise RuntimeError(f"md5 mismatch for {dest}: delete it and re-run")
            print(f"[md5 ] {name} OK")
        paths.append(dest)
    return paths


def download(cfg):
    if cfg["data"]["source"] == "synthetic":
        print("synthetic data: nothing to download")
        return []
    return download_calo(cfg) if cfg["data"]["source"] == "calo" else download_qg(cfg)
