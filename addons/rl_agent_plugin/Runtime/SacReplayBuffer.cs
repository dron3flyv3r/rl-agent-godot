using System;

namespace RlAgentPlugin.Runtime;

internal sealed class SacReplayBuffer
{
    private readonly Transition[] _buffer;
    private int _head;
    private int _count;

    public SacReplayBuffer(int capacity)
    {
        _buffer = new Transition[capacity];
    }

    public int Count => _count;
    public int Capacity => _buffer.Length;

    public void Add(Transition transition)
    {
        _buffer[_head] = transition;
        _head = (_head + 1) % _buffer.Length;
        if (_count < _buffer.Length)
        {
            _count++;
        }
    }

    public Transition[] SampleBatch(int batchSize, Random rng)
    {
        var actualBatch = Math.Min(batchSize, _count);
        var batch = new Transition[actualBatch];

        // Fisher-Yates shuffle on indices for sampling without replacement
        var indices = new int[_count];
        for (var i = 0; i < _count; i++)
        {
            indices[i] = i;
        }

        for (var i = 0; i < actualBatch; i++)
        {
            var j = i + rng.Next(_count - i);
            (indices[i], indices[j]) = (indices[j], indices[i]);
            batch[i] = _buffer[indices[i]];
        }

        return batch;
    }
}
