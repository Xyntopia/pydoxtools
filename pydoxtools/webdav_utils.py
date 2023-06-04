from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import logging
import subprocess
import time

import requests
from webdav3.client import Client

logger = logging.getLogger(__name__)


def rclone_single_sync_models(
        hostname,
        token,
        syncpath,
        method,
        file_name="",
        reversed=False
):
    """
    implements the following command for nextcloud shared webdav folders

    rclone {method} :webdav:{file_name} /local/sync/dir/{file_name} \
        --webdav-url=https://sync.example.net/public.php/webdav \
        --webdav-vendor=nextcloud \
        --webdav-user=KwkyKj8LgFZy8mo  #name of shared folder link in nextcloud
    """
    # install rclone
    # for non-shared folders it is /remote.php/webdav:
    flag = '--resync' if method == 'bisync' else ""
    if reversed:
        src = f":webdav:{file_name}"
        dest = f"{syncpath}/{file_name}"
    else:
        src = f"{syncpath}/{file_name}"
        dest = f":webdav:"

    cmd = (f"rclone {method} {src} {dest} {flag} "
           f"--webdav-url={hostname}/public.php/webdav "
           f"--webdav-vendor=nextcloud "
           f"--webdav-user={token} ")

    logger.info(f"sync {syncpath} using rclone to {hostname}")
    result = subprocess.run(cmd, shell=True, capture_output=True)
    if result.stderr:
        logger.info(result.stderr)
    return result, cmd


def continuous_sync_models():
    """sync every 60s"""
    while True:
        rclone_single_sync()
        print("sync successfully completed...")
        time.sleep(60)


# this upload command works:
# curl -u "ckzRbLYQRsBnSPr:" -T text_blockclassifier.ckpt https://www.somenextcloudurl.com/public.php/webdav/txtblk2.cpkt
def upload_file_to_public_share(hostname: str, filename: str, token: str):
    """uploads a file to a public webdav share such as a nextcloud public share

    parameter examples:

    hostname: "https://www.somenextcloudurl.com"
    filename: "/home/user/myfile.zip"
    token:  the last part of the nextcloud share "public link" is the token:
            in https://www.somenextcloudurl.com/s/P97oYZQ71doZ6BQ
            the token would be:   P97oYZQ71doZ6BQ
    """
    with open(filename, 'rb') as fo:
        url = hostname + "/public.php/webdav" + filename
        requests.put(url, auth=(token, ''), data=fo)


def push_dir_diff(hostname: str, token: str, syncpath: str) -> None:
    """push the missing files from a directory to a public share of a
    nextcloud webdav server

    url: "https://www.somenextcloudurl.com"
    syncpath: "/home/localdir/user/somedir"
    token:  the last part of the nextcloud share "public link" is the token:
            in https://www.somenextcloudurl.com/s/P97oYZQ71doZ6BQ
            the token would be:   P97oYZQ71doZ6BQ

    """
    options = {
        'webdav_hostname': hostname,
        'webdav_login': token,
        'webdav_password': "  ",
        'webdav_root': '/public.php/webdav/'
    }
    client = Client(options)
    # client.verify = False # To not check SSL certificates (Default = True)
    # client.list()
    client.push(local_directory=syncpath, remote_directory="/")


if __name__ == "__main__":
    res, cmd = rclone_single_sync()
    print(res.stderr.decode())
    print(res.stdout.decode())
