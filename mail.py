import os
import requests
import subprocess


def print_logbook(msg):
    printer = "areslog"
    title = "MSK IPC Autonomous Accelerator"
    with subprocess.Popen(["lpr","-#","3","-l","-T",title.encode("ascii"),"-P",printer,"-"], stdin=subprocess.PIPE) as lp:
        lp.stdin.write(msg.encode("utf-8"))
        

def print_logbook_http(title, msg):
    # r1 = requests.get("https://ttfinfo.desy.de/elogbookManager/Manager",
    #                  params={"name": "SINBAD-ARESelog",
    #                          "fill": "/2021/47/25.11"})
    # print(r1.url)
    # print(r1.status_code)
    r1b = requests.get("https://ttfinfo.desy.de/elogbookManager/Manager?name=SINBAD-ARESelog&fill=/2021/47/25.11")
    print(r1b.url)
    print(r1b.status_code)
    # r2 = requests.post("https://ttfinfo.desy.de/elog/FileEdit",
    #                   params={"source": "/SINBAD-ARESelog/data/2021/47/25.11/2021-11-25T14:49:50-00.xml"},
    #                   data={
    #     "author": "Jan Kaiser",
    #     "severity": "NONE",
    #     "date": "25.11.2021",
    #     "time": "14:49:51",
    #     "keywords": "not set",
    #     "location": "not set",
    #     "title": title,
    #     "text": msg,
    #     "category": "USERLOG",
    #     "metainfo": "2021-11-25T14:49:50-00.xml",
    #     "topic": "All",
    #     "expertlist": "Dinter, Hannes",
    #     "expert": "Dinter",
    #     "email": "hannes.dinter@desy.de",
    #     "femail": ""
    # })
    # print(r2.url)
    # print(r2.status_code)
    r2 = requests.post("https://ttfinfo.desy.de/elog/FileEdit?source=/SINBAD-ARESelog/data/2021/47/25.11/2021-11-25T14:49:50-00.xml",
                      data={
        "author": "Jan Kaiser",
        "severity": "NONE",
        "date": "25.11.2021",
        "time": "14:49:51",
        "keywords": "not set",
        "location": "not set",
        "title": title,
        "text": msg,
        "category": "USERLOG",
        "metainfo": "2021-11-25T14:49:50-00.xml",
        "backlink": "https://ttfinfo.desy.de/elog/XMLlist?file=/SINBAD-ARESelog/data/2021/47/25.11&xsl=/elogbook/xsl/elog.xsl",
        "topic": "All",
        "expertlist": "Dinter, Hannes",
        "expert": "Dinter",
        "email": "hannes.dinter@desy.de",
        "femail": ""
    })
    print(r2.url)
    print(r2.status_code)
        
        
def send_mail(msg, to):
    if isinstance(to, list):
        for recepient in to:
            send_mail(msg, recepient)
    else:
        os.popen(f"echo -n \"Subject: {msg}\" | sendmail {to}")

 
def main():
    print_logbook_http("MSK-IPC Autonomus Accelerator", "This is a test")
    

if __name__ == "__main__":
    main()
        

        
# https://ttfinfo.desy.de/elogbookManager/Manager?name=SINBAD-ARESelog&fill=/2021/47/25.11
# https://ttfinfo.desy.de/elog/FileEdit?source=/SINBAD-ARESelog/data/2021/47/25.11/2021-11-25T14:49:50-00.xml
# -----------------------------109536439420067959791812922022
# Content-Disposition: form-data; name="author"

# Author
# -----------------------------109536439420067959791812922022
# Content-Disposition: form-data; name="severity"

# NONE
# -----------------------------109536439420067959791812922022
# Content-Disposition: form-data; name="date"

# 25.11.2021
# -----------------------------109536439420067959791812922022
# Content-Disposition: form-data; name="time"

# 14:49:50
# -----------------------------109536439420067959791812922022
# Content-Disposition: form-data; name="keywords"

# not set
# -----------------------------109536439420067959791812922022
# Content-Disposition: form-data; name="location"

# not set
# -----------------------------109536439420067959791812922022
# Content-Disposition: form-data; name="title"

# Tittel
# -----------------------------109536439420067959791812922022
# Content-Disposition: form-data; name="text"

# Texxt
# -----------------------------109536439420067959791812922022
# Content-Disposition: form-data; name="category"

# USERLOG
# -----------------------------109536439420067959791812922022
# Content-Disposition: form-data; name="metainfo"

# 2021-11-25T14:49:50-00.xml
# -----------------------------109536439420067959791812922022
# Content-Disposition: form-data; name="backlink"

# https://ttfinfo.desy.de/elog/XMLlist?file=/SINBAD-ARESelog/data/2021/47/25.11&xsl=/elogbook/xsl/elog.xsl
# -----------------------------109536439420067959791812922022
# Content-Disposition: form-data; name="image"; filename=""
# Content-Type: application/octet-stream


# -----------------------------109536439420067959791812922022
# Content-Disposition: form-data; name="topic"

# All
# -----------------------------109536439420067959791812922022
# Content-Disposition: form-data; name="expertlist"

# -----
# -----------------------------109536439420067959791812922022
# Content-Disposition: form-data; name="experts"


# -----------------------------109536439420067959791812922022
# Content-Disposition: form-data; name="email"


# -----------------------------109536439420067959791812922022
# Content-Disposition: form-data; name="femail"


# -----------------------------109536439420067959791812922022--