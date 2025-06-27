# Environment Variables

List of environment variables used to configure DataChain behavior.

### Core Configuration

- `DATACHAIN_ROOT_DIR` – Specifies the root directory where DataChain will create the `.datachain` folder to store its internal data. (default: the current working directory).
- `DATACHAIN_SYSTEM_CONFIG_DIR` – Overrides the system-wide configuration directory (default depends on the platform).
- `DATACHAIN_GLOBAL_CONFIG_DIR` – Overrides the user's global configuration directory (default depends on the platform).
- `DATACHAIN_NO_ANALYTICS` – Disables telemetry.

### Studio Integration

- `DATACHAIN_STUDIO_URL` – Custom Studio URL.
- `DATACHAIN_STUDIO_TOKEN` – Authentication token for Studio.
- `DATACHAIN_STUDIO_TEAM` – Studio team name.

### Namespaces and projects
- `DATACHAIN_NAMESPACE` – Namespace name to use as default.
- `DATACHAIN_PROJECT` – Project name or combination of namespace name and project name separated by `.` to use as default, example: `DATACHAIN_PROJECT=dev.analytics`

Note: Some environment variables are used internally and may not be documented here. For the most up-to-date list, refer to the source code.
