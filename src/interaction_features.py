import time


class InteractionTracker:

    def __init__(self):
        self.last_interaction_time = None
        self.interaction_times = []

        self.total_edits = 0
        self.total_revisions = 0
        self.total_suggestions = 0
        self.accepted_suggestions = 0

    def record_interaction(
        self,
        prompt,
        response_time,
        edits=0,
        revisions=0,
        suggestions=0,
        accepted_suggestions=0
    ):

        now = time.time()

        # Time since previous interaction
        if self.last_interaction_time is None:
            time_since_last = 0
        else:
            time_since_last = now - self.last_interaction_time

        self.last_interaction_time = now
        self.interaction_times.append(now)

        self.total_edits += edits
        self.total_revisions += revisions
        self.total_suggestions += suggestions
        self.accepted_suggestions += accepted_suggestions

        prompt_length = len(prompt.split())

        interaction_frequency = (
            len(self.interaction_times) /
            max((now - self.interaction_times[0]) / 60, 1 / 60)
        )

        revision_rate = (
            self.total_revisions /
            max(len(self.interaction_times), 1)
        )

        suggestion_rate = (
            self.accepted_suggestions /
            max(self.total_suggestions, 1)
        )

        return {
            "response_time_sec": response_time,
            "prompt_length_words": prompt_length,
            "edit_count": self.total_edits,
            "revision_rate": revision_rate,
            "interaction_frequency": interaction_frequency,
            "time_since_last_interaction_sec": time_since_last,
            "AI_suggestion_interaction_rate": suggestion_rate
        }


# Test
tracker = InteractionTracker()

features = tracker.record_interaction(
    prompt="Explain binary lifting",
    response_time=4.2,
    edits=2,
    revisions=1,
    suggestions=3,
    accepted_suggestions=2
)

print(features)