#!/usr/bin/env python3

import socket
import threading
import logging
from struct import Struct

logger = logging.getLogger(__name__)

class Receiver(threading.Thread):
    def __init__(self, port, callback_map, unparsed_callback=None):
        threading.Thread.__init__(self, daemon=True, name=__name__)
        self.port = port
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.serversocket.bind(("0.0.0.0", port))
        self.callback_map = {Struct(format):callback for format, callback in callback_map.items()}
        self.unparsed_callback = unparsed_callback
        self.stop_event = threading.Event()

    def parse_packet(self, data):
        for struct, callback in self.callback_map.items():
            if struct.size == len(data):
                try:
                    unpacked = struct.unpack(data)
                    callback(unpacked)
                except Exception as e:
                    logger.exception(e)
                finally:
                    return
        if self.unparsed_callback is None:
            raise AttributeError("Cannot parse packet of size {}b".format(data.len))
        else:
            self.unparsed_callback(data)


    def client_loop(self, client_socket):
        while not self.stop_event.is_set():
            try:
                data = client_socket.recv(8192)
                if data:
                    logger.debug("Incoming {} b packet".format(len(data)))
                    self.parse_packet(data)
                else:
                    client_socket.close()
                    break
            except (OSError, ConnectionResetError):
                print('Connection lost')
                break
            except KeyboardInterrupt:
                return

    def run(self):
        logger.info("Starting tcp receiver thread")
        self.serversocket.listen(5)
        while not self.stop_event.is_set():
            logger.info("Waiting for connection on port {}".format(self.port))
            try:
                clientsocket, addr = self.serversocket.accept()      
                logger.info("Got a connection from {}".format(addr))
                self.client_loop(clientsocket)
            except Exception as e:
                logging.exception(e)

    def stop(self, wait=None):
        self.stop_event.set()
        self.join(wait)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    r = Receiver(4862,
        {
            "56H": lambda data: print("56H", data),
            "5f": lambda data: print("5f", data),
            "f": lambda data: print("f", data),
        },
        lambda data: print('unparsed:', data)
    )

    r.start()
    try:
        r.join()
    except KeyboardInterrupt:
        logger.info("KeyboardInterupt, bye")