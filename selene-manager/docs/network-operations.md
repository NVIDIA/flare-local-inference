# Serving instances on other hosts

Instances on the manager's host talk to it over its Unix socket and move bytes through
`/dev/shm/selene`. Instances on other hosts reach the same manager over TCP. There is one
manager per pool (see [operating.md](operating.md)); there is no manager-to-manager protocol, and
the manager is never put behind a load balancer.

## Addresses and startup warnings

Set `--prefix-cache-pool ADDRESS` on each instance:

| Address | Connection |
|---|---|
| `unix:PATH` | Same-host Unix socket and shared memory; trusts filesystem permissions. |
| `HOST:PORT` | Plain TCP (the default network mode); use a trusted network. |
| `tls:HOST:PORT` | TLS; verifies the manager against the host CA bundle. |

Ports are 1 through 65535. IPv6 hosts use brackets, for example
`[::1]:9200` or `tls:[::1]:9200`. `--prefix-cache-pool-ca FILE` replaces the
host CA bundle and requires TLS. Unix clients work with either manager network mode.
TLS uses the first existing file among `/etc/ssl/certs/ca-certificates.crt`,
`/etc/pki/tls/certs/ca-bundle.crt`, `/etc/ssl/ca-bundle.pem`, `/etc/ssl/cert.pem`
and `/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem`. These paths are fixed;
use the CA option when the bundle lives elsewhere. There are no client certificates.

At startup the manager reports network listening as
`selene: warning: network listening enabled on ADDR (TLS|plaintext, data on ADDR)`.
With `--rdma` it also reports `selene: warning: RDMA enabled (experimental)`.
RDMA bytes are unencrypted, so combining RDMA and TLS refuses startup. TLS
requires both certificate and key; missing files or a mismatched pair also fail startup.
For TCP clients, `Hello.encryption` must match the selected mode; a mismatch
or a certificate failure disables the pool for that process with one diagnostic,
without downgrading. A missing manager at instance startup produces one warning
after the startup wait, then local tiers only; see
[pool attachment](../../docs/prefix-cache-offload.md#215-sharing-through-a-pool).

## Turning it on

```sh
selene --name pool --root /var/lib/selene --memory 16G --nvme 1T --staging 1G \
  --listen 0.0.0.0:9200 --listen-data 0.0.0.0:9201 --tls-cert server.pem --tls-key server-key.pem
```

- `--listen ADDR:PORT` serves the pool protocol on that address in addition to the Unix socket.
- `--listen-data ADDR:PORT` is where those instances connect to copy bytes (see below). It
  defaults to the host of `--listen` on the next port (and to any free port when `--listen` names
  port 0). The address a grant names is the one the listener bound; when it is the unspecified
  address (`0.0.0.0`), the instance connects to the host it reached the manager by. Publish this
  port as you publish the first.
- `--tls-cert` and `--tls-key` give the manager a TLS identity. Instances select
  the matching connection mode from the address table above.
- Keep the key out of the image and mount it read-only (`deploy/compose-network.yaml`).
  The compose example publishes control port 9200 and data port 9201, and uses the host PID
  namespace so peer-process checks can see local instances.

A session on this listener has no shared memory. Instance-sourced candidates are
offered through the manager's relay; the manager's own records are offered as
level 5 (remote NVMe), as [protocol decision 4](../PROTOCOL.md#decisions) specifies.

## The data channel

Hello, sessions, notices, resync, lookup, and disk-recovery readiness work over the network
listener exactly as on the Unix socket. An instance without shared memory moves record bytes on
the **data channel**: a `Claim` or `PutBegin` in mode `DATA_CHANNEL` is granted a single-use token
and the data listener's address instead of windows. The instance connects (with TLS when the
manager has a certificate), presents the token and says which carrier moves the payload.

- **A claim** (the manager's own records, DRAM or NVMe): the store fills its windows as for a
  local requester, and the manager's handler writes each one to the instance with the checksums
  stored with the record, then `END`. The instance verifies every page and the row before it sends
  `COPY_OK`.
- **A put**: the store's windows are offered to the instance as `GRANT`s and filled from what it
  writes back; the manager checks each fill against its grant before reading the payload into the
  window. A commit that reaches the session before the last fill reaches the channel waits for it.
- **Fencing**: a transfer's windows go back to the store only after its handler has exited and
  its connection is closed (RDMA: its queue pair moved to the error state and destroyed). A stopped handler
  retains its windows until its exit is observed; readiness `quarantined_bytes` includes these
  windows while the handler is draining. A remote instance
  never touches the manager's windows, so its own exit is not waited for.
- **Limits**: a transfer that makes no progress for the claim's progress timeout (3 s by
  default) expires; a socket that stalls for 20 s fails. The metrics carry
  `selene_pool_channel_windows_total` and `selene_pool_channel_bytes_total` by `carrier`.

A neighbour's host or GPU records are offered to a session without shared memory too, and an
instance without shared memory can be the neighbour: the manager relays the windows between the two
legs through its staging, so the copy costs two network hops (or one, when either end is on the
manager's host) and uses the manager's staging for the windows in flight. The framing is
`flare_selene::shared::frame`, documented at the top of that file.

## Certificates

Generate a CA and a server certificate whose subject alternative names contain every name or
address clients dial:

```sh
openssl req -x509 -newkey rsa:3072 -nodes -days 3650 \
  -keyout ca-key.pem -out ca.pem -subj '/CN=selene-ca'
openssl req -newkey rsa:3072 -nodes -keyout server-key.pem -out server.csr -subj '/CN=pool'
printf 'subjectAltName=DNS:pool.example\n' > server.ext
openssl x509 -req -in server.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial \
  -days 3650 -extfile server.ext -out server.pem
```

Hand `ca.pem` to the instances (their pool CA setting) unless the host already trusts the CA,
and `server.pem` and `server-key.pem` to the manager.

## RDMA (experimental)

The default build and image never link libibverbs. Build with `--features rdma` (or use
`Dockerfile.rdma`) and start with `--rdma`; the
[startup warnings and TLS restriction](#addresses-and-startup-warnings) apply. The container needs
`--device /dev/infiniband` and `--ulimit memlock=-1:-1`. The manager opens the first device with
an active port and registers its regions once for local access only; `--rdma` on a host with none
refuses to start. The manager advertises no memory keys or region addresses. The instance lends
only its private transfer staging, so it cannot use RDMA to address the manager arena.

The manager does every RDMA transfer, one-sided, on a queue pair made for the transfer: it writes
unsliced claim windows into private slots the instance lends (`CREDIT`). Puts and rank legs
use TCP. The instance must be built with the
`rdma` feature and have a device; otherwise, or after one RDMA transfer fails, it asks for the
stream carrier on its connection, so a mixed pool works. Each end advertises a RoCE v2 GID, preferring
a routable (IPv4-mapped or global) one over a link-local one, because two ends must use the same RoCE
version and a device's GID index 0 is v1 on some NICs and v2 on others (Soft-RoCE); the path MTU is
1024. Hardware RoCE and Soft-RoCE interoperate on that basis, but a Soft-RoCE end that receives writes
from a hardware NIC drops packets it cannot keep up with and the transfer collapses to a few MiB/s;
use Soft-RoCE only for functional tests, or as the sending end. For a Soft-RoCE test host:

```sh
sudo modprobe rdma_rxe
sudo rdma link add rxe0 type rxe netdev eth0   # any active Ethernet device
ibv_devices                                    # rxe0 should appear
```

## Troubleshooting

| Symptom | Check |
|---|---|
| Remote instances run on local tiers only | The address and firewall; whether the instance trusts the manager's CA (the host's CA bundle or its CA file), and the one-line error it logged at start; `Hello.encryption`. |
| `DATA_CHANNEL` claims or puts refused | The manager has no network listener (`INVALID_CLAIM`, or `PutRefused{INVALID}` saying so). |
| Claims expire with `NO_PROGRESS` | The instance cannot reach `--listen-data` (firewall, a published port that is not the bound one), or it does not read fast enough. |
| Windows quarantined (`quarantined_bytes`) | A data handler did not exit after its transfer was stopped; the manager logs the claim. They return if the exit is seen, else at restart. |
| TLS handshake failures | The certificate's SAN against the name clients dial; that the CA the clients hold signed the server certificate. |
| `--rdma` refused | TLS options present, or a build without the `rdma` feature. |
