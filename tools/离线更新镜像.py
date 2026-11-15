"""离线为 OCI 镜像追加一层，把整改后的源码同步进镜像。

Docker 守护进程不可用时，用「追加覆盖层」而不是重建整个镜像：
保留原有 13 层不动，新增第 14 层覆盖 app/src、app/build/lib、
site-packages、app/configs 下的过时源码，再重算 manifest / config / index。

产出：dist/industrial-evolution-school-2026-11-14.tar
"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import pathlib
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_TAR = ROOT / "submission/提交版/镜像/industrial-evolution-school-2026-11-12.tar"
OUT_TAR = ROOT / "dist/industrial-evolution-school-2026-11-14.tar"
NEW_TAG = "school-2026-11-14"
MTIME = int(datetime.datetime(2026, 11, 14, 12, 0, 0, tzinfo=datetime.timezone.utc).timestamp())


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


TOP_LEVEL_FILES = {
    "app/README.md": "README.md",
    "app/pyproject.toml": "pyproject.toml",
    "app/run.sh": "run.sh",
}
REMOVED_SUFFIXES = (
    "llm/deepseek_provider.py",
    "configs/deepseek-dev.yaml",
    "configs/local-qwen.yaml",
)
FORBIDDEN_TOKENS = (b"deepseek", b"DeepSeek", b"Qwen", b"PyTRiO", "企业赛".encode())


def map_to_local(name: str) -> pathlib.Path | None:
    """镜像内路径 → 本地仓库相对路径。"""
    if name in TOP_LEVEL_FILES:
        return pathlib.Path(TOP_LEVEL_FILES[name])
    if name.startswith("app/src/"):
        return pathlib.Path("src") / name[len("app/src/") :]
    if name.startswith("app/build/lib/"):
        return pathlib.Path("src") / name[len("app/build/lib/") :]
    if "site-packages/industrial_evolution/" in name:
        tail = name.split("site-packages/industrial_evolution/", 1)[1]
        return pathlib.Path("src/industrial_evolution") / tail
    if "industrial_evolution-0.1.0.dist-info/METADATA" in name:
        return pathlib.Path("src/industrial_evolution.egg-info/PKG-INFO")
    if name.startswith("app/configs/"):
        return pathlib.Path("configs") / name[len("app/configs/") :]
    return None


def is_source_path(name: str) -> bool:
    """镜像内属于源码范围的路径（含 pyc，用于收集最终态）。"""
    return (
        name in TOP_LEVEL_FILES
        or name.startswith("app/src/")
        or name.startswith("app/build/lib/")
        or "site-packages/industrial_evolution/" in name
        or "industrial_evolution-0.1.0.dist-info/METADATA" in name
        or "industrial_evolution-0.1.0.dist-info/RECORD" in name
        or name.startswith("app/configs/")
    )


def is_code_file(name: str) -> bool:
    """允许写入覆盖层的源码文件；排除本机 pyc。"""
    return (
        not name.endswith(".pyc")
        and "__pycache__" not in name
        and (
            name.endswith((".py", ".yaml", ".yml", ".txt", ".typed", ".md", ".sh", ".toml", ".bat"))
            or name.endswith(("PKG-INFO", "METADATA", "RECORD"))
        )
    )


def normalize(data: bytes, name: str) -> bytes:
    """Linux 镜像内统一 LF 行尾，避免 Windows CRLF 造成全量伪差异。"""
    if name.endswith((".py", ".yaml", ".yml", ".txt", ".typed", ".md", ".sh", ".toml", "PKG-INFO", "METADATA")):
        return data.replace(b"\r\n", b"\n")
    return data


def add_blob(out: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    info.mtime = MTIME
    out.addfile(info, io.BytesIO(data))


def main() -> None:
    with tarfile.open(SRC_TAR) as t:
        index = json.loads(t.extractfile("index.json").read())
        oci_layout = t.extractfile("oci-layout").read()
        manifest_json = json.loads(t.extractfile("manifest.json").read())
        first = index["manifests"][0]
        manifest = json.loads(t.extractfile("blobs/sha256/" + first["digest"].split(":")[1]).read())
        config = json.loads(t.extractfile("blobs/sha256/" + manifest["config"]["digest"].split(":")[1]).read())

        blobs: dict[str, bytes] = {}
        for member in t.getmembers():
            if member.isfile() and member.name.startswith("blobs/sha256/"):
                blobs[member.name.split("/")[-1]] = t.extractfile(member).read()

        # 镜像内源码的「最终态」哈希：后层覆盖前层
        final: dict[str, str] = {}
        record_data: dict[str, bytes] = {}
        for layer in manifest["layers"]:
            digest = layer["digest"].split(":")[1]
            with tarfile.open(fileobj=io.BytesIO(blobs[digest]), mode="r:*") as lt:
                for member in lt.getmembers():
                    if member.isfile() and is_source_path(member.name):
                        content = lt.extractfile(member).read()
                        final[member.name] = sha(content)
                        if member.name.endswith("industrial_evolution-0.1.0.dist-info/RECORD"):
                            record_data[member.name] = content

    overrides: dict[str, bytes] = {}
    for name, old_hash in final.items():
        if not is_code_file(name):
            continue
        rel = map_to_local(name)
        if rel is None:
            continue
        local = ROOT / rel
        if not local.is_file():
            continue
        data = normalize(local.read_bytes(), rel.as_posix())
        if sha(data) != old_hash:
            overrides[name] = data

    # 本地新增、镜像中不存在的源码文件（按镜像实际存在的三处前缀补齐）
    site_prefixes = {
        n[: n.index("industrial_evolution/")]
        for n in final
        if "site-packages/" in n and "industrial_evolution/" in n
    }
    for local in sorted((ROOT / "src").rglob("*")):
        if not local.is_file():
            continue
        rel = local.relative_to(ROOT).as_posix()
        if not is_code_file(rel) or "egg-info" in rel or "__pycache__" in rel:
            continue
        data = normalize(local.read_bytes(), rel)
        tail = rel[len("src/") :]
        candidates = ["app/src/" + tail, "app/build/lib/" + tail]
        candidates += [prefix + tail for prefix in site_prefixes]
        for name in candidates:
            if name not in final:
                overrides[name] = data

    # 配置目录里新增/重命名的文件
    for local in sorted((ROOT / "configs").glob("*.yaml")):
        rel = local.relative_to(ROOT).as_posix()
        name = "app/" + rel
        if name not in final:
            overrides[name] = normalize(local.read_bytes(), rel)

    # 旧层里的 .pyc 与已删除的调试分支文件用 whiteout 移除
    whiteouts: list[str] = []
    for name in final:
        if name.endswith(".pyc") or name.endswith(REMOVED_SUFFIXES):
            parent, base = name.rsplit("/", 1)
            whiteouts.append(parent + "/.wh." + base)

    # 安装记录里移除已删除模块的条目，避免最终态留下旧文件名
    for name, data in record_data.items():
        kept = [
            line
            for line in data.decode("utf-8").splitlines()
            if "deepseek" not in line.lower() and "qwen" not in line.lower()
        ]
        overrides[name] = ("\n".join(kept) + "\n").encode("utf-8")

    if not overrides:
        raise SystemExit("没有需要覆盖的文件，镜像可能已经是最新")

    changed = sorted({map_to_local(n).as_posix() for n in overrides if map_to_local(n)})
    print(f"覆盖路径 {len(overrides)} 处，涉及本地文件 {len(changed)} 个：")
    for name in changed:
        print("  ", name)

    # 构造新层（GNU 格式避免长路径被 USTAR 截断）
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as lt:
        for name in sorted(overrides):
            add_blob(lt, name, overrides[name])
        for name in sorted(set(whiteouts)):
            add_blob(lt, name, b"")
    layer_bytes = buffer.getvalue()
    layer_digest = sha(layer_bytes)
    blobs[layer_digest] = layer_bytes

    # 更新 manifest
    config_desc = dict(manifest["config"])
    config["rootfs"]["diff_ids"].append("sha256:" + layer_digest)
    config["history"].append(
        {
            "created": "2026-11-14T12:00:00Z",
            "created_by": "COPY updated source (offline layer update, 2026-11-14)",
            "comment": "school-submission-source-update",
        }
    )
    config_bytes = json.dumps(config, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    config_digest = sha(config_bytes)
    blobs[config_digest] = config_bytes
    config_desc["digest"] = "sha256:" + config_digest
    config_desc["size"] = len(config_bytes)
    manifest["config"] = config_desc
    manifest["layers"].append(
        {
            "mediaType": "application/vnd.oci.image.layer.v1.tar",
            "digest": "sha256:" + layer_digest,
            "size": len(layer_bytes),
        }
    )
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    manifest_digest = sha(manifest_bytes)
    blobs[manifest_digest] = manifest_bytes

    first["digest"] = "sha256:" + manifest_digest
    first["size"] = len(manifest_bytes)
    first["annotations"]["org.opencontainers.image.ref.name"] = NEW_TAG
    first["annotations"]["io.containerd.image.name"] = f"docker.io/library/industrial-evolution:{NEW_TAG}"

    manifest_json[0]["Config"] = "blobs/sha256/" + config_digest
    manifest_json[0]["Layers"].append("blobs/sha256/" + layer_digest)
    manifest_json[0]["RepoTags"] = [f"industrial-evolution:{NEW_TAG}"]
    repositories = {"industrial-evolution": {NEW_TAG: config_digest}}

    OUT_TAR.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUT_TAR, "w", format=tarfile.GNU_FORMAT) as out:
        for digest, data in sorted(blobs.items()):
            add_blob(out, "blobs/sha256/" + digest, data)
        add_blob(out, "index.json", json.dumps(index, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        add_blob(out, "manifest.json", json.dumps(manifest_json, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        add_blob(out, "oci-layout", oci_layout)
        add_blob(out, "repositories", json.dumps(repositories, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    print()
    print("新层 digest :", layer_digest)
    print("新 config   :", config_digest)
    print("新 manifest :", manifest_digest)
    print("层数        :", len(manifest["layers"]))
    print("输出        :", OUT_TAR, OUT_TAR.stat().st_size, "bytes")

    # 完整性自检
    with tarfile.open(OUT_TAR) as chk:
        idx2 = json.loads(chk.extractfile("index.json").read())
        d = idx2["manifests"][0]["digest"].split(":")[1]
        raw = chk.extractfile("blobs/sha256/" + d).read()
        assert sha(raw) == d, "manifest digest 不匹配"
        m2 = json.loads(raw)
        c = m2["config"]["digest"].split(":")[1]
        assert sha(chk.extractfile("blobs/sha256/" + c).read()) == c, "config digest 不匹配"
        cfg2 = json.loads(chk.extractfile("blobs/sha256/" + c).read())
        assert len(cfg2["rootfs"]["diff_ids"]) == len(m2["layers"]), "diff_ids 与层数不一致"
        for layer in m2["layers"]:
            ld = layer["digest"].split(":")[1]
            data = chk.extractfile("blobs/sha256/" + ld).read()
            assert sha(data) == ld, f"层 {ld[:12]} digest 不匹配"
            assert len(data) == layer["size"], f"层 {ld[:12]} size 不匹配"
        # 最终态关键字
        final2: dict[str, bytes] = {}
        for layer in m2["layers"]:
            ld = layer["digest"].split(":")[1]
            with tarfile.open(fileobj=io.BytesIO(chk.extractfile("blobs/sha256/" + ld).read()), mode="r:*") as lt:
                for member in lt.getmembers():
                    if not member.isfile():
                        continue
                    parent, base = member.name.rsplit("/", 1)
                    if base.startswith(".wh."):
                        final2.pop(parent + "/" + base[4:], None)
                        continue
                    if is_source_path(member.name):
                        final2[member.name] = lt.extractfile(member).read()
        probes = {
            "config.py": b"label_budget_mode",
            "budget.py": b"_enforce_labels",
            "orchestrator.py": b"decisions.jsonl",
            "sklearn_adapter.py": b"score_metric",
        }
        for suffix, token in probes.items():
            hits = [n for n, data in final2.items() if n.endswith(suffix) and token in data]
            assert hits, f"镜像最终态缺少 {token.decode()}"
            print(f"  自检 {token.decode():<20} 命中 {len(hits)} 处")
        # 运行时副本一致性
        runtime = [n for n in final2 if "site-packages/industrial_evolution/config.py" in n]
        assert runtime, "site-packages 运行副本缺失"
        assert b"label_budget_mode" in final2[runtime[0]], "运行副本未更新"
        print("  自检 site-packages 运行副本已同步")
        dirty = sorted(
            n
            for n, data in final2.items()
            if (n.startswith("app/") or "industrial_evolution/" in n or "industrial_evolution-0.1.0.dist-info" in n)
            and any(token in data for token in FORBIDDEN_TOKENS)
        )
        assert not dirty, "镜像最终态仍含旧口径: " + ", ".join(dirty[:5])
        print("  自检 旧模型与阶段字眼残留 0 处")
    print("完整性自检通过")


if __name__ == "__main__":
    main()
