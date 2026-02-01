
import  json, time
from datetime import datetime, timezone
from pathlib import Path
from dataset_population_script import coerce_json, validate_conversation, build_generation_prompt, TOTAL_SAMPLES, MODEL_NAME, OUTFILE, get_output_text, client
from domains_scenarios.quotes_wishes_02_prompt import DOMAINS, SCENARIO_HINTS

def main():
    Path(OUTFILE).write_text("", encoding="utf-8")  # truncate

    # Precompute ordered (domain, scenario) list in the required order:
    ordered_pairs = []
    fallback = ["request a meeting next week to align on goals"]
    for d in DOMAINS:
        scen_list = SCENARIO_HINTS.get(d) or fallback
        for s in scen_list:
            ordered_pairs.append((d, s))

    i = 0
    # Keep cycling domain→scenarios in that order until we reach TOTAL_SAMPLES
    while i < TOTAL_SAMPLES:
        for domain, scenario in ordered_pairs:
            if i >= TOTAL_SAMPLES:
                break

            prompt = build_generation_prompt(domain, scenario)
            status = "ok"
            record = {}
            conv_id_for_print = ""

            try:
                resp = client.responses.create(model=MODEL_NAME, input=prompt)
                text = get_output_text(resp)

                try:
                    obj = coerce_json(text)
                    obj["conversation_id"] = f"conv_{str(i+1).zfill(4)}"
                    obj["domain"] = obj.get("domain", domain)
                    obj["scenario"] = obj.get("scenario", scenario)
                    obj["generated_at"] = datetime.now(timezone.utc).isoformat()
                    obj["model"] = MODEL_NAME

                    val = validate_conversation(obj)
                    obj["validation"] = val
                    status = "ok" if val["is_valid"] else "error"

                    record = {"status": status, "data": obj}
                    conv_id_for_print = obj["conversation_id"]
                except Exception as je:
                    status = "error"
                    record = {
                        "status": "error",
                        "error": f"json_parse_failed: {je.__class__.__name__}: {je}",
                        "raw_text": text[:2000],
                        "domain": domain,
                        "scenario": scenario,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "model": MODEL_NAME,
                    }

            except Exception as e:
                status = "error"
                record = {
                    "status": "error",
                    "error": f"api_call_failed: {e.__class__.__name__}: {e}",
                    "domain": domain,
                    "scenario": scenario,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "model": MODEL_NAME,
                }

            with open(OUTFILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            tag = "OK" if status == "ok" else "ERR"
            tail = f" - {conv_id_for_print}" if conv_id_for_print else ""
            print(f"[{i+1}/{TOTAL_SAMPLES}] {tag} - {domain} | {scenario}{tail}")

            i += 1
            time.sleep(0.25)

    print(f"Done. Wrote {TOTAL_SAMPLES} lines to {OUTFILE}.")


if __name__ == "__main__":
    main()
