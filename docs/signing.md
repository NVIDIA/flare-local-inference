# Production signing

Ignition signs runtime packages with Ed25519. Source builds contain no embedded
publisher profiles. Supply a public profile with `--publisher-keys`, or provision
profiles when building a distribution as described in
[runtime package protection](runtime-package-design.md#publisher-authentication-and-executable-access).

Publisher profiles verify signatures; they do not verify identity. Flare checks
identity through `publisher.json` at the root of the package's official source
repository. Execution requires first-use acceptance. Private keys never belong
in source control, packages, or publisher profiles.

## Publish the verification key

Upload the `publisher.json` that key creation wrote, unchanged, to the root of
every repository named as an official source. Flare fetches it from the default
branch on every online load and requires it to list the key under the publisher
namespace. The file is a list, so a rotated key stays valid
by keeping the previous profile in it, and removing a profile revokes that key
for every package it signed. A repository whose `publisher.json` omits the
signing key makes the package fail to load everywhere; a repository without the
file leaves the identity unverified, and users must then supply the profile with
`--publisher-keys`.

Each repository should also hold the license file its model card names. The
exporter records the license name and a link to that file from the model card
front matter, Flare shows both before acceptance, and typing the namespace
agrees to the terms at the link.

## Create and protect a key

Run key creation in your own interactive terminal:

```sh
.venv/bin/ignition keygen \
  --directory ~/.config/flare/signing/example-publisher \
  --name 'Example Publisher' --namespace example-publisher
```

This creates a random Ed25519 key encrypted as PKCS#8 PEM using the entered
passphrase, and a separate `publisher.json` containing only its public identity.
The directory must not already exist; key creation never overwrites a key.
On POSIX, the directory is mode `0700` and both files are mode `0600`.
Generating another key creates a different identity and fingerprint.

Use a unique, randomly generated passphrase stored in a password manager.
Ignition requires at least 16 UTF-8 bytes and confirms the passphrase. Enter it
only at the terminal prompt, never in chat, shell arguments, configuration, or
an environment variable. Signing also prompts in the terminal and refuses a
prompt whose input cannot be hidden.

Keep an encrypted backup of `signing-key.pem` and its public profile on separate
storage, with the passphrase recoverable separately. Verify that the backup can
be decrypted before depending on it. Losing either the key or its passphrase
prevents further signatures under that identity. File permissions and encryption
protect the stored key; they do not protect an unlocked signing process from a
compromised account or root. Sign only on a trusted host and review the package
being promoted.

For rotation, generate into a different directory and supply the replacement public
profile through `--publisher-keys` or a provisioned distribution. Previously signed
packages require their original public key; execution acceptance is specific to
the key and namespace. If a private key is
compromised, remove its identity from distributed builds and revoke its acceptance
in affected trust stores with `flare trust revoke FINGERPRINT`.

## Change the passphrase

```sh
.venv/bin/ignition key-password \
  --signing-key ~/.config/flare/signing/example-publisher/signing-key.pem
```

The command prompts for the existing passphrase, then the replacement twice.
It writes an encrypted temporary file with owner-only permissions beside the key,
flushes it to disk, and atomically replaces the key file. The key and fingerprint
stay the same; publisher profiles, Helios builds, and signed packages need no
changes. A wrong existing passphrase or mismatched confirmation leaves the file
untouched. An interruption before replacement can leave an encrypted hidden
temporary file beside the key.

Update the encrypted backup after changing the passphrase. Existing backups
still use their previous passphrase. A passphrase change does not revoke copies
of the private key; use key rotation if a copy may be compromised.

## Promote a development package

Use a distinct destination and the actual repository URL where the production
package will be published:

```sh
.venv/bin/ignition sign /data/flare-runtime/development \
  --output /data/flare-runtime/production \
  --signing-key ~/.config/flare/signing/example-publisher/signing-key.pem \
  --namespace example-publisher \
  --official-source https://huggingface.co/example-publisher/REPOSITORY
```

Replace `example-publisher` and `REPOSITORY` with your namespace and repository.
The command does not upload anything. The three publishing values can instead come from `[publishing]`
in the [configuration file](cli-design.md#signing-and-local-development); explicit
flags take precedence. It locates `development-publisher.json` beside that config
file. For a package prepared elsewhere, supply the original trusted public profile
with `--development-publisher PATH`. The development private key is unnecessary.

Signing a payload-format package verifies its source signatures and stored
payloads, then signs its deployments and ranks under the production identity
with a fresh package ID. Weight bytes, the model license the export recorded
from the model card, and prepared execution contracts are preserved.
Executable resources are authenticated, decoded, and re-encoded as
independent zstd level-3 frames; decoded bytes remain exact, while compressed
frame bytes may change.

The output uses the directory canonical form with readable companion names.
Use `ignition convert --form packed` to produce a single-file package afterward.
The payload format has no canonical size-sharded form or shard-size setting.
The internal logical-resource package layout uses a separate signing path
that writes content-addressed snapshot companions with a fixed 64 GiB file limit.

Invalid packages fail before publication, and existing destinations are never
overwritten. Weight identities remain independent of executable resources.

No checkpoint reload, compilation, graph capture, or GPU access is required.
The operation reads source payloads to verify them, including the weight bytes.
Allow disk space for the output and temporary decoded executable resources.
Temporary files reside in a private directory beside the destination and are
removed on normal completion or error. A forced kill can leave that directory;
remove it only after confirming no signing process is using it. The source
package remains intact. See [package snapshots](cli-design.md#package-snapshots)
for conversion that preserves an existing signature without a private key.

Rebuild Flare after changing embedded public profiles. A separately built or
installed executable retains the profiles it was built with.
