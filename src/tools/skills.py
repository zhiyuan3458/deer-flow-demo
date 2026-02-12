# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Skills 工具：与 skills 目录集成，支持列出、加载、创建 skill。"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Annotated, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 项目根目录 (deer-flow)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SKILLS_DIR = _PROJECT_ROOT / "skills"


def _get_skill_metadata(skill_path: Path) -> Optional[dict]:
    """从 SKILL.md 提取 frontmatter metadata。"""
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        return None
    try:
        content = skill_md.read_text(encoding="utf-8")
        match = re.search(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if not match:
            return None
        import yaml

        meta = yaml.safe_load(match.group(1))
        return {"name": meta.get("name", skill_path.name), "description": meta.get("description", "")}
    except Exception as e:
        logger.warning(f"Failed to parse {skill_md}: {e}")
        return None


@tool
def list_skills(
    include_description: Annotated[
        bool, "是否包含每个 skill 的 description，默认 True"
    ] = True,
):
    """列出 skills 目录下所有可用的 skill。用于了解当前有哪些 skill 可用，或判断用户请求是否与某 skill 相关。"""
    if not _SKILLS_DIR.exists():
        return f"Skills 目录不存在: {_SKILLS_DIR}"
    result = []
    for item in sorted(_SKILLS_DIR.iterdir()):
        if item.is_dir() and not item.name.startswith("."):
            meta = _get_skill_metadata(item)
            if meta:
                entry = {"name": meta["name"], "path": str(item)}
                if include_description:
                    entry["description"] = meta["description"]
                result.append(entry)
    if not result:
        return "未找到任何 skill。"
    return "\n\n".join(
        f"- **{r['name']}** ({r['path']})\n  {r.get('description', '')}"
        for r in result
    )


@tool
def load_skill_content(
    skill_name: Annotated[str, "Skill 名称，如 xlsx、skill-creator、docx"],
):
    """读取指定 skill 的 SKILL.md 完整内容。在需要执行该 skill 相关任务时，先调用此工具获取指南。"""
    skill_path = _SKILLS_DIR / skill_name
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        available = [d.name for d in _SKILLS_DIR.iterdir() if d.is_dir()]
        return f"Skill '{skill_name}' 不存在。可用: {available}"
    try:
        return skill_md.read_text(encoding="utf-8")
    except Exception as e:
        return f"读取失败: {e}"


@tool
def create_skill(
    skill_name: Annotated[str, "kebab-case 格式的 skill 名称，如 my-excel-helper"],
    output_path: Annotated[
        str,
        "输出目录路径，相对项目根或绝对路径。默认 skills/",
    ] = "skills",
):
    """根据 skill-creator 模板初始化新 skill。会创建 SKILL.md、scripts/、references/、assets/。创建后需人工编辑 SKILL.md 和资源文件。"""
    path = Path(output_path).resolve()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    init_script = _PROJECT_ROOT / "skills" / "skill-creator" / "scripts" / "init_skill.py"
    if not init_script.exists():
        return f"init_skill.py 不存在: {init_script}"
    try:
        proc = subprocess.run(
            [str(init_script), skill_name, "--path", str(path)],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=30,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode != 0:
            return f"创建失败 (exit {proc.returncode}):\n{out}\n{err}"
        return f"Skill 创建成功:\n{out}"
    except subprocess.TimeoutExpired:
        return "执行超时。"
    except Exception as e:
        return f"执行异常: {e}"


@tool
def run_skill_script(
    skill_name: Annotated[str, "Skill 名称，如 xlsx"],
    script_name: Annotated[str, " scripts/ 下的脚本名，如 recalc.py"],
    script_args: Annotated[
        str,
        "传递给脚本的参数，空格分隔。如 'output.xlsx 30'",
    ] = "",
):
    """执行指定 skill 的 scripts 目录下的 Python 脚本。如 xlsx 的 recalc.py。"""
    script_path = _SKILLS_DIR / skill_name / "scripts" / script_name
    if not script_path.exists():
        scripts_dir = _SKILLS_DIR / skill_name / "scripts"
        if scripts_dir.exists():
            available = list(scripts_dir.glob("*.py"))
            return f"脚本不存在: {script_path}\n可用: {[p.name for p in available]}"
        return f"Skill '{skill_name}' 没有 scripts 目录。"
    args = script_args.split() if script_args else []
    try:
        proc = subprocess.run(
            ["python", str(script_path)] + args,
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=60,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode != 0:
            return f"执行失败 (exit {proc.returncode}):\n{out}\n{err}"
        return out or "(无输出)"
    except subprocess.TimeoutExpired:
        return "执行超时。"
    except Exception as e:
        return f"执行异常: {e}"
