from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
from threading import Thread


class DummyWebSocket(WebSocket):

    def handleConnected(self):
        print(self.address, 'connected')

    def handleClose(self):
        print(self.address, 'closed')


class WebServer:
    """
    Creates a SimpleWebSocketServer on a new thread
    """

    def __init__(self):
        self.server = None
        self.server_thread = None

    def start(self):
        self.server = SimpleWebSocketServer('localhost', 1997, DummyWebSocket)
        self.server_thread = Thread(target=self.server.serveforever)
        self.server_thread.start()

    def get_all_connections(self):
        return self.server.connections.values()
