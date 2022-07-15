import os, platform
import subprocess, time
from pathlib import Path
import requests
import torch
from typing import Union, List


def gsutil_getsize(url: str = str()) -> Union[bool, int]:
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0


def attempt_to_download(file: Union[str, List[str]], repo='ultralytics/yolov5'):
    file = Path(str(file).strip().replace("'", str()).lower())
    print(f'file attempted for download: {file}')
    try:
        if not file.exists():
            try:
                response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()  # github api
                assets = [x['name'] for x in response['assets']]
                tag = response['tag_name']
            except:
                assets = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']
                tag = subprocess.check_output('git_tag', shell=True).decode().split()[-1]
            name = file.name
            if name in assets:
                msg = f'{file} missing, try downloading from https://github.com/{repo}/releases/'
                redundant = False
                try:
                    url = f'https://github.com/{repo}/releases/download/{tag}/{name}'
                    print(f'downloading {url} to {file}...'.capitalize())
                    torch.hub.download_url_to_file(url, str(file))
                    assert file.exists() and file.stat().st_size > 1E6
                except Exception as e:
                    print(f'Download error: {e}')
                    assert redundant, 'No secondary mirror'
                    url = f'https://storage.googleapis.com/{repo}/ckpt/{name}'
                    print(f'downloading {url} to {file}...'.capitalize())
                    os.system(f'curl -L {url} -o {file}')
                finally:
                    if not file.exists() or file.stat().st_size < 1E6:
                        file.unlink(missing_ok=True)    # remove partial downloads
                        print(f'ERROR: Download failure: {msg}')
                    print('')
                    return
    except FileExistsError as fe:
        raise FileNotFoundError("Error finding the url or the file. there seems to be an issue"
                                "try again later.")

def gdrive_download(id='yourIDhere', file='tmp.zip'):
    # Downloads a file from the Google Drive.
    t = time.time()
    file = Path(file)
    cookie = Path('cookie') # gdrive cookie
    print(f'downloading https://drive.google.com/uc?export=download={id} as {file}...', end=str())
    file.unlink(missing_ok=True)    # remove existing file
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Attempt file download
    out = "NUL" if platform.system() == "Windows" else '/dev/null'
    os.system(f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id}" > {out}')
    if os.path.exists('cookie'):    # large file
        s = f'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm={get_token()}&id={id}" -o {file}'
    else:
        s = f'curl -s -L -o {file} "drive.google.com/uc?export=download&id={id}'
    r = os.system(s)    # execute, capture return
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Error check
    if r != 0:
        file.unlink(missing_ok=True)    # remove partial
        print('Download error')
        return r
    # unzip if archive
    if file.suffix == '.zip':
        print('unzipping...', end='')
        os.system(f'unzip -q {file}')
        file.unlink()

    print(f'Done ({time.time() - t:.1f}s)')
    return r


def get_token(cookie='./cookie'):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return str()
