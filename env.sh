export TEST_ROOT=$PWD
export HOSTS=$(hostname)
export SIMS="scaleRTL-C scaleRTL-G"
if [[ "$_KSIM_AE_PATH_UPDATED" != "1" ]]; then
  export PATH=$TEST_ROOT/install/bin:$TEST_ROOT/third_party/tools/bin:$PATH
fi
if which vcs >/dev/null 2>/dev/null; then
  export SIMS="$SIMS vcs"
fi
export _KSIM_AE_PATH_UPDATED=1

_color_print() {
  tput setaf "$1"
  printf "%s     \t" "$2"
  tput setaf 7
  printf "%s\n" "$3"
}

_show_status() {
  if ! which $1 2>/dev/null > /dev/null; then
    _color_print 1 "$1" "is not found"
    return 1
  else
    show_path=$(which $1)
    if realpath --relative-to=$PWD $show_path 2>/dev/null >/dev/null; then
      new_show_path=./$(realpath --relative-to=$PWD $show_path)
      if [ ${#new_show_path} -le ${#show_path} ]; then
        show_path=$new_show_path
      fi
    fi
    _color_print 2 "$1" "is at $show_path"
    return 0
  fi
}

_show_ssh() {
  if ! ssh $1 true; then
    _color_print 1 "$1" "unable to connect"
    return 1
  else
    _color_print 2 "$1" "success to connect"
    return 0
  fi
}

show-status() {
  echo "Check Tools"
  tools="llc firtool firrtl g++ clang++ KaHyPar firclean timeit scaleRTL scaleRTL-opt"
  for tool in $(echo $tools); do
    _show_status $tool
  done
  echo
  _color_print 4 "Runs:" "$SIMS"
  echo
}

show-status
