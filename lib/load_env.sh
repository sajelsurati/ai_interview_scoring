# Shared .env loader. SOURCE this file, do not execute it.
#
#   source lib/load_env.sh
#   load_env /path/to/.env
#
# Parsed line by line rather than sourced: only KEY=VALUE lines matching a strict
# pattern are honored, so a stray command in the file cannot execute. Values may be
# bare, single-quoted, or double-quoted, and CRLF line endings are tolerated.
#
# Absent file is not an error -- every value it supplies is optional with a documented
# fallback or a later explicit check.

load_env() {
  local env_file=${1:-${ENV_FILE:-}}
  [ -n "$env_file" ] || return 0
  if [ ! -f "$env_file" ]; then
    echo "no .env at $env_file (continuing)"
    return 0
  fi

  # Precedence: an already-exported value WINS over the file, matching the usual
  # dotenv convention, so `ACCOUNT=other ./submit.sh ...` works as a one-off override.
  # Skips are announced -- never silently ignore a value the file supplied.
  local line key val skipped=()
  while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
    key=${line%%=*}
    val=${line#*=}
    val=${val%$'\r'}                      # tolerate CRLF
    val=${val%\"} ; val=${val#\"}         # strip surrounding double quotes
    val=${val%\'} ; val=${val#\'}         # or single quotes
    if [ -n "${!key:-}" ]; then
      skipped+=("$key")
      continue
    fi
    export "$key=$val"
  done < "$env_file"

  # Report key NAMES only -- never echo a value, these are secrets.
  echo "loaded $env_file (keys: $(grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' "$env_file" \
        | sed 's/=$//' | tr '\n' ' '))"
  if [ ${#skipped[@]} -gt 0 ]; then
    echo "  already set in the environment, file value ignored: ${skipped[*]}"
  fi

  local perms
  perms=$(stat -c %a "$env_file" 2>/dev/null || stat -f %Lp "$env_file" 2>/dev/null || echo "")
  case "$perms" in
    600|400|"") ;;
    *) echo "warning: $env_file is mode $perms; secrets should be 600." >&2
       echo "         fix with: chmod 600 $env_file" >&2 ;;
  esac
}
