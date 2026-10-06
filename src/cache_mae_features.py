"""Create format-4, role-separated standalone MAE feature caches."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from raid_experiments.fewshot import LIBERO_COMMIT, PROTOCOL, validate_goal_protocol
from raid_experiments.fewshot_train import _sha
from src.data_libero import LazyLiberoTransitionDataset
from src.mae_encoder import FEATURE_DIM, MAEImageEncoder, MAE_IMPLEMENTATION, MAE_PREPROCESSING, PREPROCESSING_VERSION

def cache_protocol_tasks(protocol_path, dataset_dir, output_dir, mae_checkpoint, libero_repo, device="cpu", batch_size=32,
                         *, execution_mode="scientific", encoder_factory=None):
    if execution_mode not in {"scientific", "synthetic"}:
        raise ValueError("cache execution_mode must be scientific or synthetic")
    if execution_mode == "scientific" and encoder_factory is not None:
        raise ValueError("custom encoders are forbidden for scientific feature caches")
    if execution_mode == "synthetic" and encoder_factory is None:
        raise ValueError("synthetic feature caches require an explicit fixture encoder")
    protocol = validate_goal_protocol(protocol_path)
    dataset_root, output, libero = (Path(p).expanduser().resolve() for p in (dataset_dir, output_dir, libero_repo))
    if not dataset_root.is_dir() or not libero.is_dir(): raise FileNotFoundError("dataset root and pinned LIBERO checkout are required")
    if _sha(Path(mae_checkpoint)) is None: raise FileNotFoundError(f"MAE checkpoint missing: {mae_checkpoint}")
    def git(*args):
        commands = [["git", "-C", str(libero), *args]]
        fallback = Path("/Users/ConstiX/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/git")
        if fallback.is_file(): commands.append([str(fallback), "-C", str(libero), *args])
        error = None
        for command in commands:
            try:
                return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
            except Exception as exc: error = exc
        raise RuntimeError(f"cannot verify pinned LIBERO source: {error}")
    revision, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    if revision != LIBERO_COMMIT or status: raise ValueError("LIBERO checkout must be clean and at the preregistered revision")
    output.mkdir(parents=True, exist_ok=True)
    encoder_initialization_started = time.perf_counter()
    encoder = (encoder_factory(mae_checkpoint, device) if encoder_factory is not None
               else MAEImageEncoder.from_checkpoint(mae_checkpoint, device))
    encoder_initialization_seconds = time.perf_counter() - encoder_initialization_started
    started, dataset_files, shard_map = time.perf_counter(), [], {}
    identity = {"execution_mode":execution_mode,"horizon":1,"future_representation":"recorded_rgb",
      "mae_checkpoint_sha256":_sha(Path(mae_checkpoint)),
      "mae_implementation":MAE_IMPLEMENTATION,
      "preprocessing":MAE_PREPROCESSING,
      "preprocessing_version":PREPROCESSING_VERSION,"action_convention":"libero_recorded_action_at_t_v1",
      "feature_dim":FEATURE_DIM,"libero_commit":revision}
    identity_sha = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    for task,spec in protocol["tasks"].items():
        hdf5=Path(spec["path"]).resolve()
        try: hdf5.relative_to(dataset_root)
        except ValueError as exc: raise ValueError(f"protocol HDF5 for {task} is outside the requested dataset root") from exc
        source_hash=_sha(hdf5)
        if source_hash != spec["source_sha256"]: raise ValueError(f"dataset identity changed for {task}")
        dataset_files.append({"task":task,"path":str(hdf5),"sha256":source_hash}); shard_map[task]={}
        for demo in map(int,spec["demo_ids"]):
            role=(next(k for k,v in spec["source"].items() if demo in v) if spec["role"]=="source"
                  else "query" if demo in spec["query"] else "support_pool")
            path=output/task/f"demo_{demo}.pt"
            if path.exists():
                payload=torch.load(path,map_location="cpu",weights_only=False)
                expected_count=int(spec["transitions"][str(demo)])
                if (payload.get("protocol_digest")!=protocol["digest"] or payload.get("task")!=task
                    or payload.get("demo_id")!=demo or payload.get("role")!=role
                    or payload.get("source_sha256")!=source_hash
                    or payload.get("transition_count")!=expected_count
                    or payload.get("encoder_identity",{}).get("identity_sha256")!=identity_sha
                    or payload.get("has_actions")!=(role!="query")
                    or payload.get("feat_t",torch.empty(0)).shape!=(expected_count,FEATURE_DIM)
                    or payload.get("feat_next",torch.empty(0)).shape!=(expected_count,FEATURE_DIM)
                    or (role!="query" and payload.get("actions_raw",torch.empty(0)).shape!=(expected_count,7))):
                    raise ValueError(f"existing cache shard conflicts with the requested immutable cache: {path}")
                shard_map[task][str(demo)]={"path":path.relative_to(output).as_posix(),"sha256":_sha(path),
                  "role":role,"has_actions":role!="query","transition_count":expected_count,
                  "source_sha256":source_hash,"encoder_identity_sha256":identity_sha}
                continue
            rows=LazyLiberoTransitionDataset(hdf5,{"action_mean":torch.zeros(7),"action_std":torch.ones(7)},
                                               demo_ids={demo},include_actions=role!="query")
            current,future,actions,transition_ids=[],[],[],[]
            for batch in DataLoader(rows,batch_size=batch_size,shuffle=False,num_workers=0):
                with torch.no_grad():
                    pair=encoder.encode_frames(torch.cat((batch["image_t"],batch["image_next"]),dim=0)).cpu().float()
                    current.append(pair[:len(batch["image_t"])])
                    future.append(pair[len(batch["image_t"]):])
                if role!="query": actions.append(batch["action_raw"].float())
                transition_ids.extend(f"{task}:{demo}:{int(t)}" for t in batch["timestep"].tolist())
            n=int(spec["transitions"][str(demo)])
            feat_t=torch.cat(current) if current else torch.empty((0,FEATURE_DIM))
            feat_next=torch.cat(future) if future else torch.empty((0,FEATURE_DIM))
            if feat_t.shape!=(n,FEATURE_DIM) or feat_next.shape!=(n,FEATURE_DIM):
                raise ValueError(f"transition count differs from frozen protocol for {task}:{demo}")
            payload={"protocol_digest":protocol["digest"],"task":task,"demo_id":demo,"role":role,
              "has_actions":role!="query","source_sha256":source_hash,"transition_count":n,
              "transition_ids":transition_ids,"feat_t":feat_t,"feat_next":feat_next,
              "encoder_identity":{**identity,"identity_sha256":identity_sha}}
            if role!="query": payload["actions_raw"]=torch.cat(actions).contiguous()
            path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(payload,path)
            shard_map[task][str(demo)]={"path":path.relative_to(output).as_posix(),"sha256":_sha(path),
              "role":role,"has_actions":role!="query","transition_count":n,"source_sha256":source_hash,
              "encoder_identity_sha256":identity_sha}
    manifest={"format":4,"protocol":PROTOCOL,"protocol_digest":protocol["digest"],
      "dataset_files":dataset_files,"demo_shards":shard_map,**identity,"identity_sha256":identity_sha,
      "measurements":{"feature_cache_wall_seconds":time.perf_counter()-started,
        "encoder_initialization_seconds":encoder_initialization_seconds,"device":str(device),"batch_size":batch_size}}
    manifest_path=output/"manifest.json"
    if manifest_path.exists(): raise FileExistsError(f"refusing to replace cache manifest: {manifest_path}")
    temp=manifest_path.with_suffix(".json.tmp"); temp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); temp.replace(manifest_path)
    return manifest

def main():
    parser=argparse.ArgumentParser()
    for flag in ("manifest","dataset-dir","feature-dir","mae-ckpt","libero-repo"): parser.add_argument("--"+flag,required=True)
    parser.add_argument("--device",default="cpu"); parser.add_argument("--batch-size",type=int,default=32)
    args=parser.parse_args()
    result=cache_protocol_tasks(args.manifest,args.dataset_dir,args.feature_dir,args.mae_ckpt,args.libero_repo,args.device,args.batch_size)
    print(json.dumps({"status":"complete","protocol_digest":result["protocol_digest"],"tasks":len(result["demo_shards"]),
                      "manifest":str(Path(args.feature_dir)/"manifest.json")},indent=2))
if __name__=="__main__": main()
