#!/usr/bin/env python3

from flask import Flask, render_template, request, make_response, jsonify, send_from_directory, send_file
import socket
import json
import time
import numpy as np
import io
from PIL import Image

from os import system, remove, listdir
from os.path import getmtime, getsize, join, isfile


TCP_PORT = 30000
VIDEOS_DIRECTORY = '../JetsonCode/experiments_data'

app = Flask(__name__)

def sendTCP(data):	
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0) # set timeout to two seconds
    s.connect( ('127.0.0.1', TCP_PORT) )
    s.send(data)
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

@app.route("/service/<action>/<name>")
def gen_restart(action, name):

    assert action == "restart" or action == "stop", "action can be only 'restart' or 'stop'"
    assert name == "imgproc" or name == "gen", "service name can be only 'imgproc' or 'gen'"

    print(system('systemctl '+action+' tb-'+name+'.service'))

    return 'OK'   

@app.route("/download_file/<path:file>")
def download_file(file):
    """Download a file."""
    return send_from_directory(VIDEOS_DIRECTORY, file, as_attachment=True)

@app.route("/delete_file/<path:file>")
def delete_file(file):
    """Delete a file."""
    remove(join(VIDEOS_DIRECTORY, file))

    return 'OK'

@app.route('/video_files')
def video_files():
    file_info = lambda f: [f, int(getsize(join(VIDEOS_DIRECTORY, f))/2**20), time.ctime(getmtime(join(VIDEOS_DIRECTORY, f)))]

    files = [file_info(f) for f in listdir(VIDEOS_DIRECTORY) if isfile(join(VIDEOS_DIRECTORY, f))]

    return jsonify({'files': files})

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

@app.route('/image')
def image():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0) # set timeout to two seconds
    s.connect( ('127.0.0.1', TCP_PORT) )
    s.send(bytes('r', 'utf-8') + b'\x00')

    img_bytes = bytes()
    amount_expected = 2048*2048
    while len(img_bytes) < amount_expected:
        img_bytes += s.recv(amount_expected - len(img_bytes))

    s.close()

    np_img = np.frombuffer(img_bytes, dtype=np.uint8)
    np_img = np.reshape(np_img, (2048, 2048))
    np_img_dwn = np_img[::4,::4]
    im = Image.fromarray(np_img_dwn)

    output = io.BytesIO()
    im.save(output, format='PNG')
    output.seek(0, 0)

    return send_file(output, mimetype='image/png', as_attachment=False)
    

@app.route("/")
def hello():
    tb_imgproc_status = system('systemctl is-active tb-imgproc') == 0
    tb_gen_status = system('systemctl is-active tb-gen') == 0
    
    return render_template('main.html', tb_imgproc=tb_imgproc_status, tb_gen=tb_gen_status)

if __name__ == "__main__":
    # Start the web server
    app.run(host='0.0.0.0', port=5000, debug=True)
