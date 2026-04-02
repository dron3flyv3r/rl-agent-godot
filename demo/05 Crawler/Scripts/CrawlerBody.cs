using Godot;

namespace RlAgentPlugin.Demo;

/// <summary>
/// Procedurally builds and manages a four-legged crawler creature made of rigid bodies and hinge joints.
///
/// CREATURE LAYOUT (top-down, creature walks in +X direction):
///
///        FL ─── Torso ─── FR
///        BL ─────────── BR
///
/// Each leg has two segments (hip + shin) connected by HingeJoint3D.
/// All joints hinge around the world Z axis, so legs swing forward/backward in the XY plane.
/// Each shin also carries a short ray sensor used to detect ground contact near the foot tip.
///
/// JOINT ORDER (matches action and observation arrays):
///   0: FL hip · 1: FL knee · 2: FR hip · 3: FR knee
///   4: BL hip · 5: BL knee · 6: BR hip · 7: BR knee
/// </summary>
public partial class CrawlerBody : Node3D
{
    // ── Geometry ───────────────────────────────────────────────────────────

    private static readonly Vector3 TorsoSpawnPos = new(0f, 0.75f, 0f);
    private static readonly Vector3 TorsoSize     = new(0.8f, 0.15f, 0.5f);

    private const float HipRadius  = 0.055f;
    private const float HipHeight  = 0.35f;
    private const float ShinRadius = 0.045f;
    private const float ShinHeight = 0.30f;
    private const float FootSensorLength = 0.12f;

    // Separate limits keep hips away from the torso and bias knees toward flexion
    // instead of hyperextension. This reduces self-intersection and makes the motion
    // space easier for the policy to explore.
    private const float HipLimitLower  = -0.80f;
    private const float HipLimitUpper  =  0.80f;
    private const float KneeLimitLower = -1.20f;
    private const float KneeLimitUpper =  0.20f;

    // Leg attachment offsets from torso center (bottom corners)
    private static readonly Vector3[] LegCorners =
    {
        new( 0.35f, -0.075f, -0.22f),  // FL (front-left)
        new( 0.35f, -0.075f,  0.22f),  // FR (front-right)
        new(-0.35f, -0.075f, -0.22f),  // BL (back-left)
        new(-0.35f, -0.075f,  0.22f),  // BR (back-right)
    };

    // ── Motor ──────────────────────────────────────────────────────────────

    private const float MotorMaxImpulse  = 5f;

    /// <summary>Actions in [-1, 1] are scaled to ±MaxJointVelocity rad/s by SetJointTargetVelocities.</summary>
    public const float MaxJointVelocity = 5f;

    // ── State ──────────────────────────────────────────────────────────────

    public RigidBody3D    Torso  { get; private set; } = null!;
    private RigidBody3D[] _hips  = new RigidBody3D[4];
    private RigidBody3D[] _shins = new RigidBody3D[4];
    private RayCast3D[]   _footSensors = new RayCast3D[4];

    // Indexed as [i*2] = hip joint, [i*2+1] = knee joint for leg i in [FL, FR, BL, BR]
    private HingeJoint3D[] _joints = new HingeJoint3D[8];

    private Transform3D[] _initialTransforms = System.Array.Empty<Transform3D>();

    // ── Lifecycle ──────────────────────────────────────────────────────────

    public override void _Ready()
    {
        BuildCreature();
        ExcludeInternalCollisions();
        _initialTransforms = CaptureTransforms();
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// <summary>
    /// Sets motor target velocity for all 8 joints.
    /// Each value should be in [-1, 1]; it is scaled to ±<see cref="MaxJointVelocity"/> rad/s.
    /// </summary>
    public void SetJointTargetVelocities(float[] targets)
    {
        for (var i = 0; i < 8; i++)
            _joints[i].SetParam(HingeJoint3D.Param.MotorTargetVelocity, targets[i] * MaxJointVelocity);
    }

    /// <summary>Returns 8 joint angles in radians (relative rotation around Z between parent and child body).</summary>
    public float[] GetJointAngles()
    {
        var angles = new float[8];
        for (var i = 0; i < 4; i++)
        {
            angles[i * 2]     = RelativeAngleZ(Torso,    _hips[i]);
            angles[i * 2 + 1] = RelativeAngleZ(_hips[i], _shins[i]);
        }
        return angles;
    }

    /// <summary>Returns 8 joint angular velocities in rad/s (relative, around Z).</summary>
    public float[] GetJointAngularVelocities()
    {
        var vels = new float[8];
        for (var i = 0; i < 4; i++)
        {
            vels[i * 2]     = _hips[i].AngularVelocity.Z  - Torso.AngularVelocity.Z;
            vels[i * 2 + 1] = _shins[i].AngularVelocity.Z - _hips[i].AngularVelocity.Z;
        }
        return vels;
    }

    public bool IsFootGrounded(int legIndex)
    {
        if ((uint)legIndex >= _footSensors.Length)
            return false;

        var sensor = _footSensors[legIndex];
        if (sensor is null)
            return false;

        sensor.ForceRaycastUpdate();
        if (!sensor.IsColliding())
            return false;

        return sensor.GetCollider() is Node collider && !IsOwnBodyNode(collider);
    }

    /// <summary>Teleports all body parts back to their spawn transforms and zeroes out velocities.</summary>
    public void Reset()
    {
        // _initialTransforms is populated in _Ready(). In Godot 4 children fire _Ready() before
        // parents, so if the Academy calls OnEpisodeBegin before CrawlerBody._Ready() has run
        // the array will be empty — skip the reset safely; the physics bodies are still at
        // their spawn positions from AddChild, so nothing bad happens.
        if (_initialTransforms.Length == 0) return;

        var bodies = AllBodies();
        for (var i = 0; i < bodies.Length; i++)
        {
            bodies[i].LinearVelocity  = Vector3.Zero;
            bodies[i].AngularVelocity = Vector3.Zero;
            bodies[i].GlobalTransform = _initialTransforms[i];
        }
        for (var i = 0; i < 8; i++)
            _joints[i].SetParam(HingeJoint3D.Param.MotorTargetVelocity, 0f);
    }

    // ── Construction ───────────────────────────────────────────────────────

    private void BuildCreature()
    {
        Torso = CreateBox("Torso", TorsoSpawnPos, TorsoSize, mass: 2f);

        for (var i = 0; i < 4; i++)
        {
            var corner   = TorsoSpawnPos + LegCorners[i];
            var hipPos   = corner + new Vector3(0f, -HipHeight  / 2f, 0f);
            var kneePos  = corner + new Vector3(0f, -HipHeight,       0f);
            var shinPos  = kneePos + new Vector3(0f, -ShinHeight / 2f, 0f);
            var legLabel = LegLabel(i);

            _hips[i]  = CreateCapsule($"{legLabel}Hip",  hipPos,  HipRadius,  HipHeight,  mass: 0.3f);
            _shins[i] = CreateCapsule($"{legLabel}Shin", shinPos, ShinRadius, ShinHeight, mass: 0.2f);

            _joints[i * 2]     = CreateHingeJoint(
                $"Joint_{legLabel}Hip", Torso, _hips[i], corner, HipLimitLower, HipLimitUpper);
            _joints[i * 2 + 1] = CreateHingeJoint(
                $"Joint_{legLabel}Knee", _hips[i], _shins[i], kneePos, KneeLimitLower, KneeLimitUpper);
            _footSensors[i] = CreateFootSensor($"{legLabel}FootSensor", _shins[i]);
        }
    }

    private void ExcludeInternalCollisions()
    {
        // Exclude only directly connected pairs. This keeps the articulated chain stable
        // without allowing distal leg parts to pass through the torso or other legs.
        for (var i = 0; i < 4; i++)
        {
            Torso.AddCollisionExceptionWith(_hips[i]);
            _hips[i].AddCollisionExceptionWith(_shins[i]);
        }
    }

    private Transform3D[] CaptureTransforms()
    {
        var bodies = AllBodies();
        var t = new Transform3D[bodies.Length];
        for (var i = 0; i < bodies.Length; i++)
            t[i] = bodies[i].GlobalTransform;
        return t;
    }

    // ── Factory helpers ────────────────────────────────────────────────────

    private RigidBody3D CreateBox(string name, Vector3 position, Vector3 size, float mass)
    {
        var body = new RigidBody3D { Name = name, Mass = mass, LinearDamp = 0.1f, AngularDamp = 0.5f };
        body.Position = position;
        AddChild(body);
        body.AddChild(new CollisionShape3D { Shape = new BoxShape3D { Size = size } });
        body.AddChild(new MeshInstance3D   { Mesh  = new BoxMesh    { Size = size } });
        return body;
    }

    private RigidBody3D CreateCapsule(string name, Vector3 position, float radius, float height, float mass)
    {
        var body = new RigidBody3D { Name = name, Mass = mass, LinearDamp = 0.1f, AngularDamp = 0.5f };
        body.Position = position;
        AddChild(body);
        body.AddChild(new CollisionShape3D { Shape = new CapsuleShape3D { Radius = radius, Height = height } });
        body.AddChild(new MeshInstance3D   { Mesh  = new CapsuleMesh   { Radius = radius, Height = height } });
        return body;
    }

    private HingeJoint3D CreateHingeJoint(
        string name,
        RigidBody3D bodyA,
        RigidBody3D bodyB,
        Vector3 position,
        float limitLower,
        float limitUpper)
    {
        var joint = new HingeJoint3D { Name = name };
        AddChild(joint);

        // Rotate -90° around Y so that the joint's local X axis aligns with world +Z.
        // HingeJoint3D rotates around its local X, so this makes the legs swing in the XY
        // plane — the forward/backward direction for locomotion in +X.
        joint.Transform = new Transform3D(new Basis(Vector3.Up, -Mathf.Pi / 2f), position);

        joint.NodeA = joint.GetPathTo(bodyA);
        joint.NodeB = joint.GetPathTo(bodyB);

        joint.SetFlag(HingeJoint3D.Flag.UseLimit, true);
        joint.SetParam(HingeJoint3D.Param.LimitLower, limitLower);
        joint.SetParam(HingeJoint3D.Param.LimitUpper, limitUpper);

        joint.SetFlag(HingeJoint3D.Flag.EnableMotor, true);
        joint.SetParam(HingeJoint3D.Param.MotorTargetVelocity, 0f);
        joint.SetParam(HingeJoint3D.Param.MotorMaxImpulse, MotorMaxImpulse);

        return joint;
    }

    private static RayCast3D CreateFootSensor(string name, RigidBody3D shin)
    {
        var sensor = new RayCast3D
        {
            Name           = name,
            Position       = new Vector3(0f, -ShinHeight * 0.5f, 0f),
            TargetPosition = new Vector3(0f, -FootSensorLength, 0f),
            Enabled        = true,
        };
        shin.AddChild(sensor);
        sensor.AddException(shin);
        return sensor;
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private RigidBody3D[] AllBodies()
    {
        var all = new RigidBody3D[9]; // torso + 4 hips + 4 shins
        all[0] = Torso;
        for (var i = 0; i < 4; i++) { all[1 + i] = _hips[i]; all[5 + i] = _shins[i]; }
        return all;
    }

    private static float RelativeAngleZ(RigidBody3D parent, RigidBody3D child)
    {
        var angle = child.GlobalRotation.Z - parent.GlobalRotation.Z;
        while (angle >  Mathf.Pi) angle -= Mathf.Tau;
        while (angle < -Mathf.Pi) angle += Mathf.Tau;
        return angle;
    }

    private bool IsOwnBodyNode(Node node)
    {
        for (Node? current = node; current is not null; current = current.GetParent())
        {
            if (current == this)
                return true;
        }

        return false;
    }

    private static string LegLabel(int i) => i switch { 0 => "FL", 1 => "FR", 2 => "BL", _ => "BR" };
}
