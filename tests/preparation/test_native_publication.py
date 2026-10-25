"""Native payload publication, authentication, and rejection of unsupported formats."""

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from helios.tests.signing import publisher, publish_legacy
from ignition.preparation import runtime_package_pb2 as proto
from ignition.preparation.package_file import combine, write_sources, read_index
from ignition.preparation.package_snapshot import convert, payload_promote
from tests.preparation.payload_fixture import publish, rank_source, verify_signatures, manifest, schema


def test_payload_publication_signs_every_body_and_conversion_preserves_signatures(tmp_path):
    signer = publisher()
    source = publish(tmp_path, signer)
    envelope = verify_signatures(source, Ed25519PublicKey.from_public_bytes(signer.public_key))
    packed = tmp_path / 'packed.safetensors'
    convert(source, packed, form='packed')
    assert manifest(packed) == envelope
    restored = tmp_path / 'restored'
    convert(packed, restored)
    assert verify_signatures(restored, Ed25519PublicKey.from_public_bytes(signer.public_key)) == envelope


def test_metadata_only_sources_cannot_select_a_legacy_publisher(tmp_path):
    source = tmp_path / 'rank.json'
    write_sources(source, proto.ModelIndex(version=4, ranks=[proto.BundleIndex()]), [])
    output = tmp_path / 'package'
    with pytest.raises(ValueError):
        combine(output, [source], [], publisher=publisher())
    assert not output.exists()


@pytest.mark.parametrize('form', ['directory', 'packed'])
def test_unsupported_legacy_conversion_preserves_input_and_requires_reexport(tmp_path, form):
    source = tmp_path / 'legacy.safetensors'
    index = proto.ModelIndex(version=4, ranks=[proto.BundleIndex()])
    publish_legacy(source, index, [], publisher=publisher())
    assert read_index(source)[0].version == 4
    with pytest.raises(ValueError, match='legacy package format is unsupported; re-export'):
        convert(source, tmp_path / 'converted', form=form)
    assert not (tmp_path / 'converted').exists()


def test_publication_failure_preserves_existing_destination(tmp_path):
    signer = publisher()
    source = rank_source(tmp_path / 'source')
    destination = tmp_path / 'occupied'
    destination.mkdir()
    (destination / 'keep').write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        combine(destination, [source], [], publisher=signer)
    assert (destination / 'keep').read_bytes() == b'keep'


def test_missing_source_fails_without_publishing(tmp_path):
    source = rank_source(tmp_path / 'source')
    (source.parent / 'weights').unlink()
    output = tmp_path / 'package'
    with pytest.raises(ValueError):
        combine(output, [source], [], publisher=publisher())
    assert not output.exists()


def test_payload_promotion_preserves_rank_and_rebuild_metadata(tmp_path):
    from tests.preparation.test_package_protection import development_publisher, development_profile
    original = development_publisher()
    index = proto.ModelIndex(version=4, rebuild=proto.RebuildManifest(
        version=1, recipe='fixture', checkpoint_repository='test/weights', checkpoint_revision='b' * 40,
        tensor_parallel=1))
    source = rank_source(tmp_path / 'source', index=index)
    output = tmp_path / 'development'
    combine(output, [source], [], publisher=original)
    before = manifest(output)
    target = publisher()
    promoted = tmp_path / 'production'
    payload_promote(output, promoted, publisher=target, profiles=[development_profile(original)],
                    accepted=original.namespace, target_accepted=target.namespace,
                    target_publisher=dict(namespace=target.namespace, official_source=target.official_source,
                                          development=False))
    after = verify_signatures(promoted, Ed25519PublicKey.from_public_bytes(target.public_key))
    assert [rank.body for rank in before.ranks] == [rank.body for rank in after.ranks]
    before_deployment = schema()('DeploymentBody').FromString(before.deployments[0].body)
    after_deployment = schema()('DeploymentBody').FromString(after.deployments[0].body)
    assert before_deployment.rebuild == after_deployment.rebuild
    assert after_deployment.rebuild.SerializeToString(deterministic=True) == index.rebuild.SerializeToString(deterministic=True)


def test_payload_promotion_carries_the_exported_license(tmp_path):
    """Signing rewrites publisher identity only; the checkpoint's recorded license survives."""
    from dataclasses import replace
    from ignition.preparation.protection import License
    from tests.preparation.test_package_protection import development_publisher, development_profile

    license_link = 'https://huggingface.co/example/model/blob/abcdef/LICENSE'
    original = replace(development_publisher(), license=License('apache-2.0', license_link))
    source = rank_source(tmp_path / 'source')
    output = tmp_path / 'development'
    combine(output, [source], [], publisher=original)
    recorded = schema()('PackageBody').FromString(manifest(output).package_body).publisher
    assert (recorded.license_name, recorded.license_link) == ('apache-2.0', license_link)
    target = publisher()
    promoted = tmp_path / 'production'
    payload_promote(output, promoted, publisher=target, profiles=[development_profile(original)],
                    accepted=original.namespace, target_accepted=target.namespace,
                    target_publisher=dict(namespace=target.namespace, official_source=target.official_source,
                                          development=False))
    envelope = verify_signatures(promoted, Ed25519PublicKey.from_public_bytes(target.public_key))
    publisher_metadata = schema()('PackageBody').FromString(envelope.package_body).publisher
    assert publisher_metadata.namespace == target.namespace
    assert publisher_metadata.official_source == target.official_source
    assert not publisher_metadata.development
    assert (publisher_metadata.license_name, publisher_metadata.license_link) == ('apache-2.0', license_link)


@pytest.mark.parametrize('operation', ['convert', 'sign'])
def test_cli_rejects_legacy_publication_without_creating_output(tmp_path, operation):
    from types import SimpleNamespace
    from ignition.cli import publishing
    from ignition.errors import UsageError

    source = tmp_path / 'legacy.safetensors'
    publish_legacy(source, proto.ModelIndex(version=4, ranks=[proto.BundleIndex()]), [], publisher=publisher())
    config = tmp_path / 'config.toml'
    config.write_text('')
    output = tmp_path / 'output'
    args = SimpleNamespace(package=source, output=output, form='directory', config=config)
    with pytest.raises(UsageError, match='legacy package format is unsupported; re-export'):
        getattr(publishing, operation)(args)
    assert not output.exists()
    assert read_index(source)[0].version == 4
