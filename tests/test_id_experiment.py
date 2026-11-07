import json
from types import SimpleNamespace
import unittest

from mini_genrec.context import Prompt, response_format
from mini_genrec.id_experiment import decode_ids, id_prompt, id_schema, TITLE_INSTRUCTION, ID_INSTRUCTION


class IdExperimentTests(unittest.TestCase):
    def test_only_output_instruction_and_schema_item_type_change(self):
        original = Prompt([{"role": "system", "content": "System"}, {"role": "user", "content":
                           "Frozen history\n" + TITLE_INSTRUCTION + '\nCandidates:\n[{"movieId": 12}]'}],
                          25, 3, 5, 2, 0, 0)
        v = SimpleNamespace(tokens=lambda _: 25, budget=100, encoding=SimpleNamespace(encode=str.split))
        transformed = id_prompt(original, v)
        self.assertEqual(transformed.messages[0], original.messages[0])
        self.assertEqual(transformed.messages[1]["content"],
                         original.messages[1]["content"].replace(TITLE_INSTRUCTION, ID_INSTRUCTION))
        self.assertEqual(transformed.history_events, original.history_events)
        schema = id_schema()
        schema["json_schema"]["schema"]["properties"]["ranking"]["items"]["type"] = "string"
        self.assertEqual(schema, response_format())

    def test_filter_preserves_order_and_does_not_backfill(self):
        case = SimpleNamespace(candidates=[1, 2, 3])
        dataset = SimpleNamespace(movies=SimpleNamespace(index={1, 2, 3, 4}))
        response = dict(refusal=False, finish_reason="stop", content=json.dumps({"ranking": [2, 4, 999, 2, 1]}))
        result = decode_ids(response, dataset, case)
        self.assertEqual(result.ranking, [2, 1])
        self.assertEqual((result.outside_candidates, result.outside_catalog, result.duplicates), (2, 1, 1))
        for invalid in (True, "2", 2.0):
            response["content"] = json.dumps({"ranking": [invalid]})
            with self.assertRaises(ValueError):
                decode_ids(response, dataset, case)
