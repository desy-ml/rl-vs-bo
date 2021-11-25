import os


def send_mail(msg, to):
    if isinstance(to, list):
        for recepient in to:
            send_mail(msg, recepient)
    else:
        os.popen(f"echo -n \"Subject: {msg}\" | sendmail {to}")
