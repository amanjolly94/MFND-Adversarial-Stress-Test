from abc import ABC, abstractmethod


class SearchMethod(ABC):

    def __call__(self, initial_result):

        result = self.perform_search(initial_result)
        # ensure that the number of queries for this GoalFunctionResult is up-to-date
        # result.num_queries = self.goal_function.num_queries
        return result

    @abstractmethod
    def perform_search(self, initial_result):
        raise NotImplementedError()

    def check_transformation_compatibility(self, transformation):

        return True

    @property
    def is_black_box(self):

        raise NotImplementedError()

    def get_victim_model(self):
        if self.is_black_box:
            raise NotImplementedError(
                "Cannot access victim model if search method is a black-box method."
            )
        else:
            return self.goal_function.model