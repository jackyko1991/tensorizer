import type { Params, ParamSpec } from "../types";
import { RichText } from "./Tex";

interface Props {
  specs: Record<string, ParamSpec>;
  values: Params;
  onChange: (values: Params) => void;
  disabled?: boolean;
}

export default function ParamControls({ specs, values, onChange, disabled }: Props) {
  const set = (name: string, value: number | string) => onChange({ ...values, [name]: value });

  return (
    <div className="grid gap-x-5 gap-y-3 sm:grid-cols-2">
      {Object.entries(specs).map(([name, spec]) => (
        <label key={name} className="block text-sm">
          <span className="flex justify-between text-gray-700">
            <span>
              <RichText text={spec.label ?? name} />
            </span>
            {spec.type !== "choice" && (
              <span className="font-mono text-xs text-gray-500">
                {values[name]}
                {spec.unit ? ` ${spec.unit}` : ""}
              </span>
            )}
          </span>
          {spec.type === "choice" ? (
            <select
              className="mt-1 w-full rounded-md border border-gray-300 bg-white px-2 py-1 text-sm"
              value={String(values[name])}
              disabled={disabled}
              onChange={(e) => set(name, e.target.value)}
            >
              {spec.options?.map((o) => (
                <option key={o}>{o}</option>
              ))}
            </select>
          ) : (
            <input
              type="range"
              className="mt-1 w-full accent-blue-600"
              min={spec.min}
              max={spec.max}
              step={spec.step ?? (spec.type === "int" ? 1 : "any")}
              value={Number(values[name])}
              disabled={disabled}
              onChange={(e) => set(name, Number(e.target.value))}
            />
          )}
        </label>
      ))}
    </div>
  );
}
