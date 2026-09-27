import time


class InteractionTracker:

    def __init__(self):
        self.last_interaction_time = None
        self.session_start_time = time.time()

        self.interaction_times = []

        self.total_edits = 0
        self.total_revisions = 0
        self.total_suggestions = 0
        self.accepted_suggestions = 0

        # Response time from the previous Gemini request
        self.last_response_time = 0.0

    def record_interaction(
        self,
        prompt,
        edits=0,
        revisions=0,
        suggestions=0,
        accepted_suggestions=0
    ):

        now = time.time()

        # Time since previous interaction
        if self.last_interaction_time is None:
            time_since_last = 0.0
        else:
            time_since_last = (
                now - self.last_interaction_time
            )

        self.last_interaction_time = now
        self.interaction_times.append(now)

        # Update counters
        self.total_edits += edits
        self.total_revisions += revisions
        self.total_suggestions += suggestions
        self.accepted_suggestions += accepted_suggestions

        # Prompt length
        prompt_length = len(prompt.split())

        # Session duration in minutes
        session_duration = (
            now - self.session_start_time
        ) / 60

        # Interactions per minute
        if session_duration > 0:
            interaction_frequency = (
                len(self.interaction_times)
                / session_duration
            )
        else:
            interaction_frequency = 0.0

        # Revision rate
        revision_rate = (
            self.total_revisions /
            max(len(self.interaction_times), 1)
        )

        # AI suggestion interaction rate
        if self.total_suggestions > 0:
            suggestion_rate = (
                self.accepted_suggestions /
                self.total_suggestions
            )
        else:
            suggestion_rate = 0.0

        return {
            "response_time_sec":
                self.last_response_time,

            "prompt_length_words":
                prompt_length,

            "edit_count":
                self.total_edits,

            "revision_rate":
                revision_rate,

            "interaction_frequency":
                interaction_frequency,

            "time_since_last_interaction_sec":
                time_since_last,

            "AI_suggestion_interaction_rate":
                suggestion_rate
        }

    def update_response_time(self, response_time):

        self.last_response_time = response_time