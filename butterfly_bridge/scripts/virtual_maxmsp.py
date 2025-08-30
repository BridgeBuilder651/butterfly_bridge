import time
import numpy as np

from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher

from butterfly_bridge.streaming.jxf import write_jxf

ip = '127.0.0.1'
port_recv = 8600
port_send = 9123
client = SimpleUDPClient(ip, port_send)

waveforms = np.load('./data/waveforms.npy')

def message_handler(address, *args):
    """A handler for incoming OSC messages."""
    print(f"Received message on '{address}': {args}")

dispatcher = Dispatcher()
dispatcher.map("/bridge", message_handler)

server = BlockingOSCUDPServer((ip, port_recv), dispatcher)

for i in range(1000):
    # generate face waveform data
    print(f'Sending waveform {i} to {ip=}, {port_send=} /butterfly')
    index = np.random.randint(0, len(waveforms))
    waveform = waveforms[index][:8192]
    write_jxf('./data/waveform.jxf', waveform, plane_count=False)
    client.send_message('/butterfly', 1)
    server.handle_request()

    time.sleep(0.3)
