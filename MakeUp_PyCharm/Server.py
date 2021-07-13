from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
import time
from urllib.parse import urlparse, parse_qs
import json
import numpy as np
from flask import Flask
import pyodbc
import ast
import pandas as pd

hostName = "127.0.0.1"
hostPort = 9007


class MyServer(BaseHTTPRequestHandler):
    app = Flask(__name__)

    def sentiment(self):
        pass
    #     do something
    # to use the model you need firefly

    def do_GET(self):
        # getparams
        query_components = parse_qs(urlparse(self.path).query)
        self.send_response(200)
        self.send_header("Content-type", 'application/json')
        self.end_headers()
        if "sentiment" in self.path:
            pass
        #     call function
        self.send_response(200)
        self.end_headers()
        json_content = json.dumps("the function/ model response", ensure_ascii=False)  # json.dumps(res)
        print(json_content)
        self.wfile.write(bytes(str(json_content), "utf-8"))
        return

# generate the server
myServer = HTTPServer((hostName, hostPort), MyServer)
print(time.asctime(), "Server Starts - %s:%s" % (hostName, hostPort))
try:
    myServer.serve_forever()
except KeyboardInterrupt:
    pass

# stop the server
myServer.server_close()
print(time.asctime(), "Server Closed - %s:%s" % (hostName, hostPort))
