#!/bin/bash
declare -A args

check_runtime_environment() {
    if ! python --version "$1" &>/dev/null; then
        echo "ERROR: Python is not available. Make sure Python is installed and venv has been activated."
        exit 1
    fi

    # Get Python version
    python_version=$(python -c 'import sys; print(sys.version_info[:])')
    IFS=', ' read -r -a values <<< "$(sed 's/[()]//g; s/,//g' <<< "$python_version")"

    # Validate that we are on a version greater than 3
    if ! [[ ${values[0]} -ge 3 ]]; then
        echo "ERROR: The current major version of python "${values[0]}" is less than required: 3"
        exit 1
    fi

    # Validate that the minor version is at least 10
    if ! [[ ${values[1]} -ge 12 ]]; then
        echo "ERROR: The current minor version of python "${values[1]}" is less than required: 12"
        exit 1
    fi

    echo "The installed python version "${values[0]}"."${values[1]}" meets the minimum requirement (3.12)."

    # Check that the required packages are installed. These should be bundled with the OS and/or Python version. 
    # If they do not exists, they should be installed manually. We do not want to install these in the run script,
    # as it could mess up the local system

    package_list=("libssl-dev" "python"${values[0]}"."${values[1]}"-dev")

    error=0
    for package_name in "${package_list[@]}"; do
        if ! dpkg -l | grep -q -w "^ii  $package_name"; then
            echo "ERROR: $package_name is not installed. Please install it manually."
            error=1
        fi
    done

    if [[ $error -eq 1 ]]; then
        exit 1
    fi

    if [ -n "$VIRTUAL_ENV" ]; then
        echo "Virtual environment is activated: $VIRTUAL_ENV"
    else
        echo "WARNING: Virtual environment is not activated. It is recommended to run this script in a python venv."
    fi
}

parse_arguments() {

    while [[ $# -gt 0 ]]; do
        if [[ $1 == "--"* ]]; then
            arg_name=${1:2}  # Remove leading "--" from the argument name

            # Special handling for logging argument
            if [[ "$arg_name" == "logging"* ]]; then
                shift
                if [[ $1 != "--"* ]]; then
                    IFS='.' read -ra parts <<< "$arg_name"
                    args[${parts[0]}]=${parts[1]}
                fi
            else
                shift
                args[$arg_name]="$1"  # Assign the argument value to the argument name
            fi
        fi
        shift
    done

    for key in "${!args[@]}"; do
        echo "Argument: $key, Value: ${args[$key]}"
    done
}

pull_repo_and_checkout_branch() {
    local branch="${args['branch']}"

    # Pull the latest repository
    git pull --all

    # Change to the specified branch if provided
    if [[ -n "$branch" ]]; then
        echo "Switching to branch: $branch"
        git checkout "$branch" || { echo "Branch '$branch' does not exist."; exit 1; }
    fi

    local current_branch=$(git symbolic-ref --short HEAD)
    git fetch &>/dev/null  # Silence output from fetch command
    if ! git rev-parse --quiet --verify "origin/$current_branch" >/dev/null; then
        echo "You are using a branch that does not exists in remote. Make sure your local branch is up-to-date with the latest version in the main branch."
    fi
}

install_packages() {
    # local cfg_version=$(grep -oP 'version\s*=\s*\K[^ ]+' setup.cfg)
    local installed_version=$(pip show SoundsRight | grep -oP 'Version:\s*\K[^ ]+')

    # Load dotenv configuration
    DOTENV_FILE=".env"
    if [ -f "$DOTENV_FILE" ]; then
        # Load environment variables from .env file
        export $(grep -v '^#' $DOTENV_FILE | xargs)
        echo "Environment variables loaded from $DOTENV_FILE"
    fi

    # if [[ "$cfg_version" == "$installed_version" ]]; then
    #     echo "Subnet versions "$cfg_version" and "$installed_version" are matching: No installation is required."
    # else
    echo "Installing python package with pip with validator extras"
    pip install -e .[validator]
    
    # fi

    # Uvloop re-implements asyncio module which breaks bittensor. It is
    # not needed by the default implementation of the
    # soundsright-subnet, so we can uninstall it.
    if pip show uvloop &>/dev/null; then
        echo "Uninstalling conflicting module uvloop"
        pip uninstall -y uvloop
    fi

    echo "All python packages are installed"
}

generate_pm2_launch_file() {
    echo "Generating PM2 launch file"
    local cwd=$(pwd)
    local neuron_script="${cwd}/soundsright/neurons/validator.py"
    local interpreter="${VIRTUAL_ENV}/bin/python"
    local branch="${args['branch']}"
    local name="${args['name']}"
    local dataset_size="${args['dataset_size']}"
    local max_memory_restart="${args['max_memory_restart']}"
    # Script arguments
    local netuid="${NETUID}"
    local subtensor_chain_endpoint="${SUBTENSOR_CHAIN_ENDPOINT}"
    local wallet_name="${WALLET}"
    local wallet_hotkey="${HOTKEY}"
    local logging_value="${LOG_LEVEL}"
    local healthcheck_api_host="$HEALTHCHECK_API_HOST"
    local healthcheck_api_port="$HEALTHCHECK_API_PORT"

    # Construct argument list for the neuron
    if [[ -z "$netuid" || -z "$wallet_name" || -z "$wallet_hotkey" || -z "$name" || -z  "$max_memory_restart" ]]; then
        echo "name, max_memory_restart, netuid, wallet.name, and wallet.hotkey are mandatory arguments."
        exit 1
    fi

    local launch_args="--netuid $netuid --wallet.name $wallet_name --wallet.hotkey $wallet_hotkey"

    if [[ -n "$subtensor_chain_endpoint" ]]; then
        launch_args+=" --subtensor.network $subtensor_chain_endpoint"
    fi

    if [[ -n "$logging_value" ]]; then
        launch_args+=" --log_level $logging_value"
    fi

    if [[ -n "$healthcheck_host" ]]; then
        launch_args+=" --healthcheck_host $healthcheck_host"
    fi

    if [[ -n "$healthcheck_port" ]]; then
        launch_args+=" --healthcheck_port $healthcheck_port"
    fi

    if [[ -n "$dataset_size" ]]; then
        dataset_size_value="${arg#*=}"
        launch_args+=" --dataset_size $dataset_size"
    fi

    if [[ -v args['debug_mode'] ]]; then
        launch_args+=" --debug_mode"
    fi

    if [[ -v args['no_sgmse'] ]]; then
        launch_args+=" --no_sgmse"
    fi

    echo "Launch arguments: $launch_args"

    cat <<EOF > ${name}.config.js
module.exports = {
    apps: [
        {
            "name"                  : "${name}",
            "script"                : "${neuron_script}",
            "interpreter"           : "${interpreter}",
            "args"                  : "${launch_args}",
            "max_memory_restart"    : "${max_memory_restart}"
        }
    ]
}
EOF
}

launch_pm2_instance() {
    local name="${args['name']}"
    eval "pm2 start ${name}.config.js"
}

echo "### START OF EXECUTION ###"
# Parse arguments and assign to associative array
parse_arguments "$@"

check_runtime_environment
echo "Python venv checks completed. Sleeping 2 seconds."
sleep 2
pull_repo_and_checkout_branch
echo "Repo pulled and branch checkout done. Sleeping 2 seconds."
sleep 2
install_packages
echo "Installation done. Sleeping 2 seconds."
sleep 2
echo "Generating PM2 ecosystem file"
generate_pm2_launch_file
echo "Launching PM instance"
launch_pm2_instance