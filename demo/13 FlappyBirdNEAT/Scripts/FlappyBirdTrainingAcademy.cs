using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo;

public partial class FlappyBirdTrainingAcademy : RLGeneticAcademy
{
    public override void OnEpisodeEnd(AcademyEpisodeEndArgs args)
    {
        base.OnEpisodeEnd(args);

        if (GetParent() is FlappyBirdController controller)
            controller.OnAgentEpisodeEnd(args);
    }
}
