#!/usr/bin/env python3

from flask import Flask, render_template, request, make_response
import socket
import json
import time

TCP_PORT = 30000

app = Flask(__name__)

def sendTCP(data):	
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0) # set timeout to two seconds
    s.connect( ('127.0.0.1', TCP_PORT) )
    s.send(data) # Am I alive command
    # rec_data = s.recv(1)
    s.close()
    # return rec_data

# @app.route('/_sendRaspiIP')
# def sendRaspiIP():
# 	global mainLogger, raspi_addr
# 	raspi_addr = request.args.get('ip', '0.0.0.0')
    
# 	print 'i'+raspi_addr

# 	sendTCP('i'+raspi_addr, IP_ADDR)
# 	mainLogger.info('Raspi address changed to {} by {}'.format(raspi_addr, request.remote_addr))

# 	return 'OK'

@app.route("/start")
def start():
    sendTCP(bytes('s', 'ascii'))
    return "Start command"

@app.route("/stop")
def stop():
    sendTCP(bytes('q', 'ascii'))
    return "Stop command"    

@app.route('/handle_data')
def handle_data():

    display_checkboxes = request.args.get('display_checkboxes').split('|')

    config = {}
    config["show"] = request.args.get('display_onoff', 'off') == "on"
    config["savevideo"] = request.args.get('savevideo_onoff', 'off') == "on"
    config["show_markers"] = 'markers' in display_checkboxes
    config["show_labels"] = 'labels' in display_checkboxes
    config["improc_thrs_G"] = int(request.args.get('img_g_thrs', '100'))
    config["improc_thrs_R"] = int(request.args.get('img_r_thrs', '140'))

    jsonconfig = json.dumps(config)

    #print(jsonconfig)
    sendTCP(bytes('o'+jsonconfig, 'utf-8'))

    # restart if needed
    restart_request = request.args.get('restart', 'without_restart') == 'with_restart'

    if restart_request:
        stop()
        time.sleep(2)
        start()
    
    return 'OK'

@app.route("/")
def hello():
    # global mainLogger
    # mainLogger.info('Main page loaded by ' + request.remote_addr)

    # return render_template('main.html', led_min = 1, led_max = 148)
    return render_template('main.html')

if __name__ == "__main__":
    # Start the web server
    app.run(host='0.0.0.0', port=5000, debug=True)
