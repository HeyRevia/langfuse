import { CodeMirrorEditor } from "@/src/components/editor";
import { usePlaygroundContext } from "../context";

const PlaygroundConfigPanel = () => {
  const { config, updateConfig } = usePlaygroundContext();

  const handleChange = (value: string) => {
    try {
      const parsed = JSON.parse(value);
      updateConfig(parsed);
    } catch (error) {
      console.error(error);
    }
  };
  return (
    <div className="flex flex-col space-y-4">
      <div className="flex flex-col space-y-2">
        <p className="text-sm font-medium">Config</p>
      </div>
      <CodeMirrorEditor
        minHeight={200}
        value={JSON.stringify(config, null, 2)}
        onChange={handleChange}
        editable
        mode="json"
      />
    </div>
  );
};

export { PlaygroundConfigPanel };
