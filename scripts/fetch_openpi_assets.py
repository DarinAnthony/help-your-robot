"""Download openpi's public base weights into its cache over HTTPS, so openpi never shells out to gsutil.

openpi fetches gs://openpi-assets/... with ``gsutil``, which can fail on Colab without a signed-in Google account. The
bucket is public: this lists it with the JSON API and downloads each object with curl into the path openpi checks
first (``$OPENPI_DATA_HOME``, default ``~/.cache/openpi``, then ``openpi-assets/<prefix>``). Standard library only.

    python3 scripts/fetch_openpi_assets.py [checkpoints/pi05_base/params ...]
"""

import json
import os
import subprocess
import sys
import urllib.request

B = 'https://storage.googleapis.com'


def fetch(prefix: str) -> None:
    home = os.path.expanduser(os.environ.get('OPENPI_DATA_HOME', '~/.cache/openpi'))
    dest = os.path.join(home, 'openpi-assets', prefix)
    if os.path.isdir(dest):
        print(f'{prefix}: already in {dest}')
        return
    items, token = [], ''
    while True:
        q = f'{B}/storage/v1/b/openpi-assets/o?prefix={prefix}/&fields=items(name,size),nextPageToken{token}'
        page = json.load(urllib.request.urlopen(q))
        items += page.get('items', [])
        if 'nextPageToken' not in page:
            break
        token = f"&pageToken={page['nextPageToken']}"
    files = [i for i in items if not i['name'].endswith('/')]
    assert files, f'nothing under gs://openpi-assets/{prefix}'
    tmp = dest + '.partial'
    jobs = []
    for i in files:
        f = os.path.join(tmp, i['name'][len(prefix) + 1:])
        os.makedirs(os.path.dirname(f), exist_ok=True)
        jobs.append(f"{f} {B}/openpi-assets/{i['name']}\n")
    subprocess.run(['xargs', '-P', '8', '-n', '2', 'curl', '-sfL', '--retry', '5', '-o'], input=''.join(jobs),
                   text=True, check=True)
    for i in files:   # every object complete
        f = os.path.join(tmp, i['name'][len(prefix) + 1:])
        assert os.path.getsize(f) == int(i['size']), f
    os.rename(tmp, dest)
    print(f"{prefix}: {len(files)} files, {sum(int(i['size']) for i in files) / 1e9:.1f} GB -> {dest}")


if __name__ == '__main__':
    for p in sys.argv[1:] or ['checkpoints/pi05_base/params']:
        fetch(p.strip('/'))
