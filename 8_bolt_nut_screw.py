"""
8_bolt_nut_screw.py ─ 볼트-너트 체결(screw-in) standalone 검증 (v2)

역할 (v1 대비 수정 — 실제 체결 방향에 맞춤):
  - 볼트: 바닥에 "고정"되어 서 있음(나사부가 하늘을 향함). 로봇이 건드리지 않는 고정 지그.
  - 너트: M0609 그리퍼가 파지 → 볼트 위로 이동/정렬 → 손목을 돌려 볼트에 내려 체결.

자산 (v1의 절차적 지오메트리 대신 Isaac Sim nucleus 정품 자산 사용):
  - factory_bolt_m8_tight.usd / factory_nut_m8_tight.usd
    (Isaac/Props/Factory — NVIDIA "Factory" 정밀조립 벤치마크용 M8 볼트/너트)
  - 실측(2026-07-20, inspect 스크립트로 확인):
      볼트: bbox z=[0, 26]mm(로컬 원점=베이스, 나사 끝=26mm 위), RigidBody 이미 적용,
            자체 root_joint(FixedJoint, body0=world)로 "고정"되도록 이미 저작되어 있음
            → 별도 지그 코드 없이 참조만 하면 바로 고정된 볼트가 됨.
      너트: bbox z=[8, 14.5]mm(원점 기준, 즉 원점은 바닥면보다 8mm 아래), RigidBody 적용되어
            있으나 mass=0/density=0 → 우리가 직접 질량을 채워야 동적 시뮬 가능.
      둘 다 collision approximation = "sdf"(실제 나사산 형상 기반 정밀 충돌, 동적 바디에도 사용 가능).
      기본 자세가 이미 수직(나사축=월드 Z)이라 별도 회전 보정 불필요.

⚠ 설계상 스코프 (v1에서 이미 검증/합의된 전제, 그대로 유지):
  A. PhysX 에는 나사(helical) 조인트가 없음 → "회전각 ↔ 하강량(피치)" 을 코드로 커플링하는
     운동학적 근사로 체결을 구동한다. 다만 이번엔 실물 나사산 SDF 충돌이 있어 실제 접촉도 함께 일어남.
  B. joint_6 가동범위는 ±360° 지만 ALIGN 단계에서 이미 일부를 소모하므로, 재파지(래칫) 없이
     1차는 부분회전(기본 0.75턴)만 검증한다(v1 실측으로 확인된 안전 범위).

제어 구조 (v1과 동일 — 검증된 구조 재사용):
  ALIGN phase : PickPlaceController(RMPFlow) 로 너트 pick + 볼트 위 정렬까지만 담당.
                event7(그리퍼 open) 진입 순간을 가로채 release 를 막고 SCREW phase 로 전환.
  SCREW phase : 별도 RMPFlowController 인스턴스로 Cartesian 목표(위치 하강+yaw 누적) 직접 구동.
                너트는 FixedJoint 로 그리퍼에 강체 용접돼 있어 EE 를 돌리면 너트가 같이 돎.
  JUDGE phase : 정지 후 너트 강체의 실측 위치/자세로 하강깊이·xy정렬·기울기 판정.

실행:
  /home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh \
      /home/rokey/cobot3_ws/isaacpjt/M0609/8_bolt_nut_screw.py
  → 뷰포트에서 Play 를 누르면 실행. Stop 후 다시 Play 하면 재실행.
  환경변수 BOLT_HEADLESS=1 이면 헤드리스로 자동 실행 후 결과를 출력하고 종료한다.
"""

import os
from isaacsim import SimulationApp

_HEADLESS = os.environ.get("BOLT_HEADLESS") == "1"
simulation_app = SimulationApp({"headless": _HEADLESS})

import sys
from pathlib import Path
import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "rmpflow"))
from m0609_pick_place_controller import PickPlaceController  # noqa: E402
from m0609_rmpflow_controller import RMPFlowController  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
#  [A] 로봇 상수 (6/7_connector_insertion*.py 와 동일 — 검증된 물리 설정 재사용)
# ══════════════════════════════════════════════════════════════════════════
ROBOT_USD_PATH   = str(_THIS_DIR / "Collected_m0609_camera/m0609_camera.usd")
ROBOT_URDF_PATH  = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
ROBOT_DESC_PATH  = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
RMPFLOW_CFG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")
ROBOT_PRIM_PATH  = "/World/m0609"
EE_LINK_NAME     = "link_6"
GRIPPER_JOINTS   = ["finger_joint", "right_inner_knuckle_joint"]
FINGER_LINKS     = ["left_inner_finger", "right_inner_finger"]

DRIVE_STIFFNESS, DRIVE_DAMPING, DRIVE_MAX_FORCE = 1.0e8, 1.0e6, 1.0e8
# GRIP_CLOSE: finger_joint 가동범위는 [0, 1.18rad](urdf 실측).
#   ⚠ 실측(2026-07-20): M8(13~15mm) 때는 손가락이 너무 안 오므려져 붕 떠보여서 가동범위
#   상한 가까이(0.95)까지 밀어붙였는데, M16(24~27.7mm)으로 바꾼 뒤 그 값을 그대로 쓰니
#   lift(ev4) 시작 직후 몇 스텝 만에 물리가 완전히 폭발함(너트가 EE와 분리되며 튕겨나감,
#   joint_vel 수천 rad/s). 관절 한계 턱밑까지 미는 건 그 자체로 불안정한 설정이라 부품
#   크기에 맞춰 다시 조정 — 6/7_connector_insertion*.py 가 36~37mm 부품에 썼던
#   0.6~0.72 범위(M16 은 그와 비슷한 크기대)로 되돌림.
GRIP_OPEN, GRIP_CLOSE, GRIP_DELTA = [0.0, 0.0], [0.68, 0.68], [-0.5, -0.5]
GRASP_JOINT = "/World/grasp_joint"

# ALIGN 단계(pick + 볼트 위 정렬) 이벤트 타이밍. event7(open) 진입은 가로채 SCREW 로 전환한다.
#   ⚠ event4(lift) 를 0.02(50step)→0.01(100step)로 늦춤 — M16 폭발이 lift 시작 직후
#   몇 스텝 안에 시작됐음(7_connector_insertion_real.py 의 "event4 는 절대 빠르게 하지
#   않는다" 교훈과 동일 — 지지면에서 떼어내는 구간은 급하게 하면 충격이 폭주함).
EVENTS_DT = [0.011, 0.006, 0.05, 0.1, 0.01, 0.013, 0.003, 1.0, 0.011, 0.08]

MAX_LIN_VEL = 3.0
MAX_ANG_VEL = 20.0
# 솔버 반복수/접촉 튜닝 — IsaacLab 공식 Factory nut_thread 태스크(정밀조립 벤치마크, 우리와 동일
#   M8 볼트/너트 자산 사용)의 factory_env_cfg.py 값을 그대로 채용:
#   https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py
#   position_iteration=192("interpenetration 방지에 중요"), velocity_iteration=1,
#   contact_offset=5mm/rest_offset=0(mm 단위 나사산 규모에 맞춘 세밀한 접촉),
#   max_depenetration_velocity=5.0(겹침 해소 시 튕겨나감 방지 — v1에서 겪은 문제와 직결).
#   PHYSICS_DT(1/60)는 그대로 유지 — RMPFlow 가 1/60 적분을 전제로 튜닝돼 있어(7_connector_
#   insertion_real.py 문서화된 교훈) IsaacLab 의 1/120 을 따르면 모션 타이밍이 깨짐.
ART_POS_ITERS, ART_VEL_ITERS = 192, 1
# ⚠ 실측(2026-07-20): IsaacLab 값(5mm)을 그대로 쓰면 너트(13~15mm)의 40%에 가까운 크기라
#   콜리전 프록시가 실제 메시보다 훨씬 크게 부풀어 "떠있는" 것처럼 보임(뷰포트 육안 확인).
#   부품 규모(mm 단위)에 맞춰 1mm로 축소.
CONTACT_OFFSET, REST_OFFSET = 0.001, 0.0
MAX_DEPENETRATION_VEL = 5.0
PHYSICS_DT = 1.0 / 60.0


# ══════════════════════════════════════════════════════════════════════════
#  [B] 볼트/너트 정품 자산 + 체결 파라미터
# ══════════════════════════════════════════════════════════════════════════
FACTORY_BOLT_REL = "/Isaac/Props/Factory/factory_bolt_m16_tight/factory_bolt_m16_tight.usd"
FACTORY_NUT_REL  = "/Isaac/Props/Factory/factory_nut_m16_tight/factory_nut_m16_tight.usd"
# ⚠ 실측(2026-07-20): M8(13~15mm)은 RG2 그리퍼 손가락 패드(원기둥형, 지름 40mm대)에 비해
#   너무 작아서 아무리 오므려도 패드 사이에 붕 떠있는 것처럼 보임(스크린샷 확인) — 콜리전
#   오프셋 문제가 아니라 그리퍼-부품 스케일 자체의 미스매치였음. M16(볼트 24x24x41mm,
#   너트 24x27.7x13mm — M8의 약 2배)으로 교체해 그리퍼 패드와 비율을 맞춘다.
#   M20(65mm 볼트)은 세로로 너무 길어서 제외.

# 볼트: 바닥에 고정(자산 자체의 root_joint 가 body0=world 로 이미 고정 저작되어 있음).
BOLT_XY     = np.array([0.45, -0.20])   # 볼트 고정 위치
BOLT_BASE_Z = 0.0                       # 참조 위치 = 볼트 로컬 원점(=베이스, 실측과 일치)
BOLT_LEN    = 0.041                     # 실측(M16): 베이스(0) ~ 나사 끝(41mm)
BOLT_TIP_Z  = BOLT_BASE_Z + BOLT_LEN

# 너트: 동적 강체로 참조 + 질량 직접 설정(자산 자체는 mass=0).
NUT_PICK_XY          = np.array([0.45, 0.20])   # 너트 초기 대기 위치
NUT_ORIGIN_TO_BOTTOM = 0.016    # 실측(M16): 너트 로컬 원점 → 바닥면까지 오프셋(16mm)
NUT_HEIGHT           = 0.013    # 실측(M16): 너트 높이(13mm)
NUT_MASS             = 0.015    # 자산 mass=0 → 직접 설정(M16 스틸너트 근사, 15g)
CANON_QUAT = np.array([1.0, 0.0, 0.0, 0.0])   # 두 자산 모두 기본자세가 이미 수직 정렬(identity)

# 받침대(pedestal): 너트를 바닥이 아니라 살짝 띄운 대 위에 둔다.
#   ⚠ 실측(2026-07-20): 너트를 바닥(z=0)에 그대로 두면 파지 하강 시 손가락이 바닥과 충돌해
#   자세가 무너짐(스크린샷으로 확인) — 그리퍼가 내려갈 여유 공간을 만들어준다.
PEDESTAL_HEIGHT = 0.020   # 20mm
PEDESTAL_SIZE   = 0.045   # 45x45mm 발판(M16 너트 24x27.7mm 보다 넉넉하게)

# 체결(SCREW) 커플링 — 나사산 SDF 접촉은 있으나, 실제 구동은 회전각↔하강량 커플링으로 한다(이슈 A).
#   ⚠ 실측(2026-07-20): 실제 M8 피치(1.25mm)를 그대로 쓰면 SCREW_TURNS=0.75 에서 총 하강량이
#   0.94mm 뿐이라, ALIGN 여유(수mm)조차 다 못 줄이고 계속 볼트 위에 "떠있는" 것처럼 보임
#   (판정 수치는 통과해도 육안으론 체결로 안 보임). 검증 스크립트 목적(실제 나사산 물리 대신
#   회전↔하강 커플링 근사, 이슈 A)에 맞춰 하강 계수를 시각적으로 확인 가능한 수준으로 키운다
#   (실제 M8 피치 값이 아님 — 검증용 근사 계수).
SCREW_DESCENT_PER_TURN_M = 0.020   # 회전 1턴(360°)당 하강량(가상, 시각적 체결감 확보용)
SCREW_TURNS       = 0.75      # 목표 회전수(이슈 B: 1차는 재파지 없이 부분회전만 검증 — v1 실측으로
                              #   joint_6 가 ALIGN 자세에서 이미 일부 회전을 소모한 상태라 2턴 요구 시
                              #   1바퀴를 살짝 넘는 지점에서 가동범위 한계에 부딪혀 IK 가 요동쳤음.
                              #   0.75턴은 그 한계 이전에서 안전하게 끝나는 값).
SCREW_OMEGA_DEG_S = 120.0     # 각속도(도/초)
SCREW_DIRECTION   = -1.0      # 조임 방향 부호(오른나사, 위에서 볼 때 시계 = world +Z축 음의 yaw)
SCREW_HOVER_CLEAR = 0.001     # ALIGN 종료 시 너트 바닥면 ↔ 볼트 끝 간 여유(최소화 — 급강하 방지 목적만)

# 파지점(그립점) = 너트 로컬 원점 기준 오프셋. PICK/PLACE 둘 다 "그립점" 기준 좌표를 쓴다
#   (6_connector_insertion.py 의 PICK_POS/PLACE_POS 관례와 동일).
NUT_GRASP_Z_LOCAL = NUT_ORIGIN_TO_BOTTOM + NUT_HEIGHT / 2.0   # 너트 몸통 중앙(≈11.25mm)

# 너트 초기 위치: 받침대 위(바닥면이 받침대 상단에 닿도록)
NUT_REST_ROOT_Z = PEDESTAL_HEIGHT - NUT_ORIGIN_TO_BOTTOM
PICK_POS = np.array([NUT_PICK_XY[0], NUT_PICK_XY[1], NUT_REST_ROOT_Z + NUT_GRASP_Z_LOCAL])

# ALIGN 목표: 너트 바닥면이 볼트 끝보다 SCREW_HOVER_CLEAR 만큼 위에 오도록.
NUT_ALIGN_ROOT_Z = (BOLT_TIP_Z + SCREW_HOVER_CLEAR) - NUT_ORIGIN_TO_BOTTOM
PLACE_POS = np.array([BOLT_XY[0], BOLT_XY[1], NUT_ALIGN_ROOT_Z + NUT_GRASP_Z_LOCAL])
EE_OFFSET = np.array([0.0, 0.0, 0.20])

MAX_HEADLESS_STEPS = 6000


# ══════════════════════════════════════════════════════════════════════════
#  [C] 헬퍼 (find_prim_path / set_all_drives / solver_iters_only / filter_pair /
#            make_grasp_joint / engage_grasp / release_grasp 는
#            7_connector_insertion_real.py 와 동일 — 검증된 물리 설정 그대로 재사용)
# ══════════════════════════════════════════════════════════════════════════
def find_prim_path(root_path, name):
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def set_all_drives(root_path):
    stage = omni.usd.get_context().get_stage()
    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        for dt in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI.Get(prim, dt)
            if drive:
                drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                drive.GetDampingAttr().Set(DRIVE_DAMPING)
                drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                n += 1
    print(f"  [OK] 드라이브 {n}개 균일 설정 (stiffness={DRIVE_STIFFNESS:.0e})")


def solver_iters_only(root_path):
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    art_prim = None
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_prim = p
            break
    art_prim = art_prim or root
    ap = PhysxSchema.PhysxArticulationAPI.Apply(art_prim)
    ap.CreateSolverPositionIterationCountAttr(ART_POS_ITERS)
    ap.CreateSolverVelocityIterationCountAttr(ART_VEL_ITERS)
    ap.CreateEnabledSelfCollisionsAttr(False)
    print(f"  [OK] 솔버 반복수 상향: pos={ART_POS_ITERS} vel={ART_VEL_ITERS}, selfCol=off")


def tune_physics_scene():
    """PhysicsScene 레벨 튜닝(IsaacLab Factory nut_thread 값 채용) — 씬 전체 솔버/접촉/GPU 설정.
       World() 가 자동 생성한 PhysicsScene prim 을 찾아서 적용(경로를 하드코딩하지 않고 스캔)."""
    stage = omni.usd.get_context().get_stage()
    scene_prim = None
    for p in Usd.PrimRange(stage.GetPrimAtPath("/")):
        if p.HasAPI(UsdPhysics.Scene) or p.GetTypeName() == "PhysicsScene":
            scene_prim = p
            break
    if scene_prim is None:
        print("  [경고] PhysicsScene prim을 못 찾음 — 씬 레벨 튜닝 생략")
        return
    ps = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    ps.CreateSolverTypeAttr("TGS")
    ps.CreateMaxPositionIterationCountAttr(ART_POS_ITERS)
    ps.CreateMaxVelocityIterationCountAttr(ART_VEL_ITERS)
    ps.CreateBounceThresholdAttr(0.2)
    ps.CreateFrictionOffsetThresholdAttr(0.01)
    ps.CreateFrictionCorrelationDistanceAttr(0.00625)
    ps.CreateEnableGPUDynamicsAttr(True)     # SDF 메시 콜라이더(볼트/너트)는 GPU 파이프라인 필요
    ps.CreateBroadphaseTypeAttr("GPU")
    ps.CreateGpuMaxRigidContactCountAttr(2 ** 23)
    ps.CreateGpuMaxRigidPatchCountAttr(2 ** 23)
    ps.CreateGpuCollisionStackSizeAttr(2 ** 28)
    ps.CreateGpuMaxNumPartitionsAttr(1)      # IsaacLab 주석: "안정적 시뮬레이션에 중요"
    print(f"  [OK] PhysicsScene 튜닝: solverType=TGS maxPosIter={ART_POS_ITERS} GPU dynamics on")


def filter_pair(prim_path_a, prim_path_b):
    stage = omni.usd.get_context().get_stage()
    a = stage.GetPrimAtPath(prim_path_a)
    UsdPhysics.FilteredPairsAPI.Apply(a).CreateFilteredPairsRel().AddTarget(prim_path_b)


def make_grasp_joint(body0_path, body1_path):
    stage = omni.usd.get_context().get_stage()
    j = UsdPhysics.FixedJoint.Define(stage, GRASP_JOINT)
    j.CreateBody0Rel().SetTargets([body0_path])
    j.CreateBody1Rel().SetTargets([body1_path])
    j.CreateJointEnabledAttr(False)


def _quatf(q):
    return Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def axis_tilt_deg(quat_wxyz):
    """물체 로컬 +Z 축이 world +Z 축과 이루는 각(도). 자기 축(체결) 회전에는 영향받지 않는다
       (회전대칭 물체가 "빙글빙글 돌고 있음"을 "넘어짐"으로 오판하지 않기 위해 사용)."""
    w, x, y, z = [float(v) for v in quat_wxyz]
    rot = Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    local_z = rot.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
    cos_a = float(np.clip(local_z[2], -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def yaw_rotated_quat(base_wxyz, delta_deg):
    """base_wxyz 를 world +Z 축 기준 delta_deg 만큼 추가 회전시킨 쿼터니언(w,x,y,z).
       SCREW phase 목표 자세 계산에 사용(체결 회전 누적)."""
    base_q = Gf.Quatd(float(base_wxyz[0]),
                       Gf.Vec3d(float(base_wxyz[1]), float(base_wxyz[2]), float(base_wxyz[3])))
    base_rot = Gf.Rotation(base_q)
    extra_rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(delta_deg))
    combined = extra_rot * base_rot
    q = combined.GetQuat()
    return np.array([q.GetReal(), *q.GetImaginary()])


def engage_grasp(p0, q0, p1, q1):
    stage = omni.usd.get_context().get_stage()
    m0 = Gf.Matrix4d(); m0.SetRotateOnly(Gf.Quatd(_quatf(q0))); m0.SetTranslateOnly(Gf.Vec3d(float(p0[0]), float(p0[1]), float(p0[2])))
    m1 = Gf.Matrix4d(); m1.SetRotateOnly(Gf.Quatd(_quatf(q1))); m1.SetTranslateOnly(Gf.Vec3d(float(p1[0]), float(p1[1]), float(p1[2])))
    rel = m1 * m0.GetInverse()
    t = rel.ExtractTranslation(); r = rel.ExtractRotationQuat()
    j = UsdPhysics.FixedJoint.Get(stage, GRASP_JOINT)
    j.CreateLocalPos0Attr(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
    j.CreateLocalRot0Attr(Gf.Quatf(float(r.GetReal()), *[float(x) for x in r.GetImaginary()]))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.GetJointEnabledAttr().Set(True)


def release_grasp():
    stage = omni.usd.get_context().get_stage()
    j = UsdPhysics.FixedJoint.Get(stage, GRASP_JOINT)
    if j:
        j.GetJointEnabledAttr().Set(False)


def initialize_robot(robot, world):
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )


# ── 볼트/너트 정품 자산 참조 ─────────────────────────────────────────────────
def _deinstance(prim_path):
    """visuals/collisions 가 instanceable 참조라 기본 PrimRange 순회에 안 잡힘 —
       7_connector_insertion_real.py 의 _meshes_under() 와 동일하게 해제."""
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    for _ in range(6):
        changed = False
        for p in Usd.PrimRange(root):
            if p.IsInstanceable():
                p.SetInstanceable(False)
                changed = True
        if not changed:
            break


def _apply_contact_offsets(root_path):
    """collision 적용된 Mesh 들에 mm 스케일 나사산에 맞는 작은 contact/rest offset 적용
       (IsaacLab Factory 값: contact=5mm, rest=0 — 기본값(수cm대)은 우리 규모엔 너무 큼)."""
    stage = omni.usd.get_context().get_stage()
    n = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if p.GetTypeName() == "Mesh" and UsdPhysics.CollisionAPI(p):
            pc = PhysxSchema.PhysxCollisionAPI.Apply(p)
            pc.CreateContactOffsetAttr(CONTACT_OFFSET)
            pc.CreateRestOffsetAttr(REST_OFFSET)
            n += 1
    return n


def build_pedestal(world, xy, material):
    """너트를 바닥이 아니라 이 위에 올려둔다 — 파지 하강 시 손가락이 바닥과 충돌하지 않도록."""
    cx, cy = float(xy[0]), float(xy[1])
    c = world.scene.add(FixedCuboid(
        prim_path="/World/pedestal", name="pedestal",
        position=np.array([cx, cy, PEDESTAL_HEIGHT / 2.0]),
        scale=np.array([PEDESTAL_SIZE, PEDESTAL_SIZE, PEDESTAL_HEIGHT]),
        color=np.array([0.4, 0.4, 0.42]),
    ))
    c.apply_physics_material(material)
    wall = omni.usd.get_context().get_stage().GetPrimAtPath("/World/pedestal")
    UsdPhysics.FilteredPairsAPI.Apply(wall).CreateFilteredPairsRel().AddTarget(ROBOT_PRIM_PATH)
    print(f"  [OK] 받침대 @ ({cx},{cy}) {PEDESTAL_SIZE*1000:.0f}x{PEDESTAL_SIZE*1000:.0f}mm"
          f" h{PEDESTAL_HEIGHT*1000:.0f}mm (로봇충돌 필터)")


def reference_bolt(bolt_url):
    """볼트: 참조만 하면 자체 root_joint(FixedJoint, body0=world)로 고정됨 — 추가 지그 불필요."""
    stage = omni.usd.get_context().get_stage()
    root_path = "/World/bolt"
    UsdGeom.Xform.Define(stage, root_path)
    add_reference_to_stage(bolt_url, root_path + "/geo")
    xf = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(root_path))
    xf.SetTranslate(Gf.Vec3d(float(BOLT_XY[0]), float(BOLT_XY[1]), float(BOLT_BASE_Z)))
    _deinstance(root_path)
    _apply_contact_offsets(root_path)
    print(f"  [OK] 볼트(정품자산, 고정) @ ({BOLT_XY[0]},{BOLT_XY[1]},{BOLT_BASE_Z}) 길이 {BOLT_LEN*1000:.0f}mm")


def reference_nut(nut_url, material):
    """너트: 동적 강체로 참조 + 질량 직접 설정(자산 자체 mass=0)."""
    stage = omni.usd.get_context().get_stage()
    root_path = "/World/nut"
    UsdGeom.Xform.Define(stage, root_path)
    add_reference_to_stage(nut_url, root_path + "/geo")
    xf = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(root_path))
    xf.SetTranslate(Gf.Vec3d(float(NUT_PICK_XY[0]), float(NUT_PICK_XY[1]), float(NUT_REST_ROOT_Z)))
    _deinstance(root_path)

    body_path = f"{root_path}/geo/factory_nut"
    body_prim = stage.GetPrimAtPath(body_path)
    if not body_prim.IsValid():
        raise RuntimeError(f"너트 RigidBody prim을 못 찾음: {body_path}")
    mass_api = UsdPhysics.MassAPI.Apply(body_prim)
    mass_api.CreateMassAttr(NUT_MASS)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(body_prim)
    rb.CreateMaxLinearVelocityAttr(MAX_LIN_VEL)
    rb.CreateMaxAngularVelocityAttr(MAX_ANG_VEL)
    rb.CreateMaxDepenetrationVelocityAttr(MAX_DEPENETRATION_VEL)   # 겹침 해소 시 튕겨나감 방지
    rb.CreateSolverPositionIterationCountAttr(ART_POS_ITERS)
    rb.CreateSolverVelocityIterationCountAttr(ART_VEL_ITERS)

    for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if p.GetTypeName() == "Mesh" and UsdPhysics.CollisionAPI(p):
            SingleGeometryPrim(prim_path=str(p.GetPath()),
                                name=p.GetName() + "_g").apply_physics_material(material)
    n_off = _apply_contact_offsets(root_path)

    print(f"  [OK] 너트(정품자산, 동적) @ ({NUT_PICK_XY[0]},{NUT_PICK_XY[1]},{NUT_REST_ROOT_Z:.4f})"
          f" mass={NUT_MASS*1000:.0f}g contactOffset적용={n_off}개")
    return SingleRigidPrim(body_path, name="nut")


# ══════════════════════════════════════════════════════════════════════════
#  [D] 메인
# ══════════════════════════════════════════════════════════════════════════
def main():
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()

    print("\n[1] 로봇 USD 로드")
    stage = omni.usd.get_context().get_stage()
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    world_prim.GetReferences().AddReference(ROBOT_USD_PATH)
    for _ in range(15):
        simulation_app.update()
    print("  [OK] robot")

    print("\n[2] 물리 설정")
    set_all_drives(ROBOT_PRIM_PATH)
    solver_iters_only(ROBOT_PRIM_PATH)
    tune_physics_scene()

    print("\n[3] 볼트/너트 정품자산 로드")
    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("assets_root 를 못 찾음 — 네트워크/nucleus 설정 확인 필요")
    part_mat = PhysicsMaterial(  # 마찰값 IsaacLab Factory 기준(1.0/1.0)에 맞춤
        prim_path="/World/Physics_Materials/part_mat",
        static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
    )
    reference_bolt(assets_root + FACTORY_BOLT_REL)
    build_pedestal(world, NUT_PICK_XY, part_mat)
    nut = reference_nut(assets_root + FACTORY_NUT_REL, part_mat)
    filter_pair("/World/nut", ROBOT_PRIM_PATH)   # 손가락↔너트 짜냄 폭발 방지(파지는 FixedJoint 담당)

    print("\n[4] 로봇 등록")
    ee_path = find_prim_path(ROBOT_PRIM_PATH, EE_LINK_NAME)
    if ee_path is None:
        raise RuntimeError(f"'{EE_LINK_NAME}' 링크를 찾을 수 없음")
    gripper = ParallelGripper(
        end_effector_prim_path=ee_path, joint_prim_names=GRIPPER_JOINTS,
        joint_opened_positions=np.array(GRIP_OPEN),
        joint_closed_positions=np.array(GRIP_CLOSE),
        action_deltas=np.array(GRIP_DELTA),
    )
    robot = world.scene.add(SingleManipulator(
        prim_path=ROBOT_PRIM_PATH, name="m0609_robot",
        end_effector_prim_path=ee_path, gripper=gripper,
    ))
    finger_mat = PhysicsMaterial(
        prim_path="/World/Physics_Materials/finger_mat",
        static_friction=1.8, dynamic_friction=1.4, restitution=0.0,
    )
    for ln in FINGER_LINKS:
        lp = find_prim_path(ROBOT_PRIM_PATH, ln)
        if lp:
            SingleGeometryPrim(prim_path=lp, name=f"{ln}_geom").apply_physics_material(finger_mat)
    print(f"  [OK] SingleManipulator, EE={ee_path}")

    make_grasp_joint(ee_path, str(nut.prim_path))

    world.reset()
    initialize_robot(robot, world)
    for _ in range(30):
        world.step(render=True)

    print("\n[5] 컨트롤러 생성 (ALIGN=PickPlaceController, SCREW=RMPFlowController)")
    align_controller = PickPlaceController(
        name="m0609_nut_align_controller",
        gripper=robot.gripper, robot_articulation=robot,
        end_effector_initial_height=0.30, events_dt=EVENTS_DT,
        urdf_path=ROBOT_URDF_PATH, robot_description_path=ROBOT_DESC_PATH,
        rmpflow_config_path=RMPFLOW_CFG_PATH, end_effector_frame_name=EE_LINK_NAME,
    )
    screw_controller = RMPFlowController(
        name="m0609_screw_cspace_controller", robot_articulation=robot,
        urdf_path=ROBOT_URDF_PATH, robot_description_path=ROBOT_DESC_PATH,
        rmpflow_config_path=RMPFLOW_CFG_PATH, end_effector_frame_name=EE_LINK_NAME,
    )
    print(f"  [OK] PICK {np.round(PICK_POS,3)}  PLACE(정렬) {np.round(PLACE_POS,3)}")
    print(f"       체결목표: {SCREW_TURNS}턴, 하강계수 {SCREW_DESCENT_PER_TURN_M*1000:.1f}mm/턴,"
          f" 총 하강목표 {SCREW_TURNS*SCREW_DESCENT_PER_TURN_M*1000:.2f}mm")

    # ── 상태머신: ALIGN → SCREW → JUDGE → DONE ─────────────────────────────
    phase = {"name": "ALIGN", "attached": False, "theta_deg": 0.0,
             "start_pos": None, "start_quat": None, "start_nut_z": None, "reported": False,
             "place_pos_run": PLACE_POS.copy()}

    def reset_cycle():
        world.reset()
        initialize_robot(robot, world)
        for _ in range(30):
            world.step(render=True)
        align_controller.reset()
        screw_controller.reset()
        release_grasp()
        phase.update(name="ALIGN", attached=False, theta_deg=0.0,
                     start_pos=None, start_quat=None, start_nut_z=None, reported=False,
                     place_pos_run=PLACE_POS.copy())
        phase.pop("_last_ev", None)
        np0, nq0 = nut.get_world_pose()
        print(f"  [DBG] 정착 후(그립 전) nut_pos={np.round(np.asarray(np0),3)}"
              f" tilt={axis_tilt_deg(np.asarray(nq0)):.1f}deg")

    def step_cycle():
        """한 물리 스텝만큼 상태머신을 전진시킨다."""
        if phase["name"] == "ALIGN":
            ev = align_controller.get_current_event()
            phase["_align_step"] = phase.get("_align_step", 0) + 1
            if ev != phase.get("_last_ev", -1):
                np_dbg, nq_dbg = nut.get_world_pose()
                tilt_dbg = axis_tilt_deg(np.asarray(nq_dbg))
                print(f"  [DBG] ev={ev} nut_pos={np.round(np.asarray(np_dbg),3)} tilt={tilt_dbg:.1f}deg")
                phase["_last_ev"] = ev
            if phase["attached"] and ev in (3, 4, 5) and phase["_align_step"] % 3 == 0:
                np_dbg, nq_dbg = nut.get_world_pose()
                eep_dbg, _ = robot.end_effector.get_world_pose()
                nv = nut.get_linear_velocity()
                nvmax = float(np.max(np.abs(np.asarray(nv)))) if nv is not None else -1.0
                print(f"  [DBG-LIFT] step={phase['_align_step']} ev={ev}"
                      f" nut={np.round(np.asarray(np_dbg),3)} EE={np.round(np.asarray(eep_dbg),3)}"
                      f" tilt={axis_tilt_deg(np.asarray(nq_dbg)):.1f} max|nv|={nvmax:.2f}")
            if 3 <= ev < 7 and not phase["attached"]:
                eep_, eeq_ = robot.end_effector.get_world_pose()
                np_, _ = nut.get_world_pose()
                engage_grasp(eep_, eeq_, np_, CANON_QUAT)
                phase["attached"] = True
                # 파지 순간 실측 EE-너트(root) z오프셋으로 PLACE 목표를 보정 —
                #   (nominal EE_OFFSET+GRASP_Z_LOCAL 가정과 실제 그리퍼 기하 사이의 오차를 상쇄.
                #   7_connector_insertion_real.py 의 grasp_xy_drift 보정과 동일 발상)
                grasp_z_offset = float(np.asarray(eep_)[2] - np.asarray(np_)[2])
                corrected_place_z = NUT_ALIGN_ROOT_Z + grasp_z_offset - EE_OFFSET[2]
                phase["place_pos_run"] = PLACE_POS.copy()
                phase["place_pos_run"][2] = corrected_place_z
                print(f"  [DBG] 파지 순간 EE={np.round(np.asarray(eep_),3)} nut={np.round(np.asarray(np_),3)}"
                      f" z오프셋={grasp_z_offset*1000:.1f}mm → 보정PLACE.z={corrected_place_z*1000:.1f}mm")
            if ev >= 7:
                # PickPlaceController 가 그리퍼를 열려는 시점 — release 대신 SCREW 로 전환.
                sp, sq = robot.end_effector.get_world_pose()
                phase["start_pos"] = np.asarray(sp).copy()
                phase["start_quat"] = np.asarray(sq).copy()
                np_, _ = nut.get_world_pose()
                phase["start_nut_z"] = float(np.asarray(np_)[2])
                phase["name"] = "SCREW"
                print(f"  [ALIGN 완료] EE={np.round(phase['start_pos'],3)} → SCREW 시작")
                return
            action = align_controller.forward(
                picking_position=PICK_POS, placing_position=phase["place_pos_run"],
                current_joint_positions=robot.get_joint_positions(),
                end_effector_offset=EE_OFFSET,
            )
            robot.apply_action(action)

        elif phase["name"] == "SCREW":
            phase["_screw_step"] = phase.get("_screw_step", 0) + 1
            phase["theta_deg"] += SCREW_OMEGA_DEG_S * PHYSICS_DT
            done = phase["theta_deg"] >= SCREW_TURNS * 360.0
            theta = min(phase["theta_deg"], SCREW_TURNS * 360.0)
            frac = theta / 360.0
            target_pos = phase["start_pos"].copy()
            target_pos[2] = phase["start_pos"][2] - frac * SCREW_DESCENT_PER_TURN_M
            target_quat = yaw_rotated_quat(phase["start_quat"], SCREW_DIRECTION * theta)
            action = screw_controller.forward(
                target_end_effector_position=target_pos,
                target_end_effector_orientation=target_quat,
            )
            robot.apply_action(action)
            if phase["_screw_step"] % 15 == 0 or phase["_screw_step"] <= 3:
                np_dbg, nq_dbg = nut.get_world_pose()
                eep_dbg, _ = robot.end_effector.get_world_pose()
                tilt_dbg = axis_tilt_deg(np.asarray(nq_dbg))
                jv = robot.get_joint_velocities()
                mjv = float(np.max(np.abs(np.asarray(jv)))) if jv is not None else -1.0
                print(f"  [DBG-SCREW] step={phase['_screw_step']} theta={theta:.0f} "
                      f"target_z={target_pos[2]*1000:.1f}mm nut={np.round(np.asarray(np_dbg),3)} "
                      f"tilt={tilt_dbg:.1f} EE_z={float(np.asarray(eep_dbg)[2])*1000:.1f}mm max|jv|={mjv:.1f}")
            if done:
                phase["name"] = "SETTLE"
                phase["settle_steps"] = 0
                print(f"  [SCREW 완료] 누적회전={phase['theta_deg']:.0f}° → 정착 대기")

        elif phase["name"] == "SETTLE":
            phase["settle_steps"] = phase.get("settle_steps", 0) + 1
            if phase["settle_steps"] >= 20:
                phase["name"] = "JUDGE"

        elif phase["name"] == "JUDGE" and not phase["reported"]:
            release_grasp()
            pos, quat = nut.get_world_pose()
            pos = np.asarray(pos); quat = np.asarray(quat)
            descent_mm = (phase["start_nut_z"] - float(pos[2])) * 1000.0
            # 체결 깊이 판정 = "볼트 끝(나사 시작점)보다 너트 바닥면이 얼마나 아래로 내려갔는가"(mm).
            #   ⚠ 실측(2026-07-20): target_z 는 계속 하강을 명령해도(SCREW_TURNS*DESCENT_PER_TURN
            #   전량) 실제 너트는 그만큼 못 따라가고 중간에 멈춤(예: 15mm 명령 → 실제 8.6mm) —
            #   이건 실물 나사산 SDF 접촉이 진짜 저항으로 작용해 물리적으로 막힌 것(정상/오히려
            #   바람직한 신호). 그래서 "명령 대비 몇 %를 따라갔는가" 대신 "볼트 끝 기준 절대
            #   체결 깊이"로 판정한다.
            nut_bottom_z = float(pos[2]) + NUT_ORIGIN_TO_BOTTOM
            engagement_mm = (BOLT_TIP_Z - nut_bottom_z) * 1000.0
            xy_err = float(np.linalg.norm(pos[:2] - BOLT_XY))
            tilt = axis_tilt_deg(quat)
            success = (engagement_mm >= 3.0) and (xy_err < 0.010) and (tilt < 15.0)
            print("\n" + "=" * 60)
            print(f"[결과] 너트 최종 위치 = {np.round(pos, 4)}")
            print(f"       회전 누적 = {phase['theta_deg']:.0f}° (목표 {SCREW_TURNS*360:.0f}°)")
            print(f"       하강 이동량 = {descent_mm:.2f}mm (명령량 대비 — 접촉저항으로 일부만 진행 가능)")
            print(f"       체결 깊이(볼트 끝 기준) = {engagement_mm:.2f}mm")
            print(f"       볼트 xy 오차 = {xy_err*1000:.1f}mm,  기울기 = {tilt:.1f}도")
            print(f"       체결 성공 = {success}")
            print(f"       (실물 나사산 SDF 접촉이 저항으로 작동 — 구동은 회전/하강 커플링 근사)")
            print("=" * 60 + "\n")
            phase["reported"] = True

    # ── 실행: GUI(Play 엣지) / 헤드리스(자동) ──────────────────────────────
    if _HEADLESS:
        print("\n[6] 헤드리스 자동 실행\n")
        reset_cycle()
        for _ in range(MAX_HEADLESS_STEPS):
            step_cycle()
            world.step(render=False)
            if phase["name"] == "JUDGE" and phase["reported"]:
                break
        simulation_app.close()
        return

    print("\n[6] 준비 완료 — 뷰포트에서 Play 를 누르면 체결을 실행합니다.")
    print("     (Stop 후 다시 Play 하면 처음부터 재실행)\n")
    was_playing = False
    while simulation_app.is_running():
        world.step(render=True)
        playing = world.is_playing()

        if playing and not was_playing:
            print("[Play] 체결 시퀀스 시작")
            reset_cycle()

        if playing:
            step_cycle()

        was_playing = playing

    simulation_app.close()


if __name__ == "__main__":
    main()
