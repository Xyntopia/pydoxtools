import requests
from webdav3.client import Client


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
