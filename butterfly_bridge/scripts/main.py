import time
import numpy as np

from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher

from butterfly_bridge.clustering.clustering import Clustering

clustering = Clustering(
    epsilon=200,
    lambd=0.00001,
    beta=0.6,
    mu=2,
    min_samples=1,
)

ip = '127.0.0.1'
port_recv = 9123
port_send = 8600
client = SimpleUDPClient(ip, port_send)


def message_handler(address, *args):
    """A handler for incoming OSC messages."""
    print(f"Received message on '{address}': {args}")

    #label = clustering.add_sample_to_clustering(np.random.rand(20))

    client.send_message('/bridge', 6)


dispatcher = Dispatcher()
dispatcher.map("/butterfly", message_handler, client,clustering)

server = BlockingOSCUDPServer((ip, port_recv), dispatcher)
server.serve_forever()
