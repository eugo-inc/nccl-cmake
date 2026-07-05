/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "param.h"
#include "debug.h"
#include "env.h"

#include <algorithm>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <mutex>
#include <pwd.h>
#include "os.h"

// @EUGO_CHANGE: @begin:
// See `EUGO_CHANGE` below!
#include <filesystem>
#include <cstring>
// @EUGO_CHANGE: @end


const char* userHomeDir() {
  struct passwd *pwUser = getpwuid(getuid());
  return pwUser == NULL ? NULL : pwUser->pw_dir;
}

void setEnvFile(const char* fileName) {
  FILE * file = fopen(fileName, "r");
  if (file == NULL) return;

  char *line = NULL;
  char envVar[1024];
  char envValue[1024];
  size_t n = 0;
  ssize_t read;
  while ((read = getline(&line, &n, file)) != -1) {
    if (line[0] == '#') continue;
    if (line[read-1] == '\n') line[read-1] = '\0';
    int s=0; // Env Var Size
    while (line[s] != '\0' && line[s] != '=') s++;
    if (line[s] == '\0') continue;
    strncpy(envVar, line, std::min(1023,s));
    envVar[std::min(1023,s)] = '\0';
    s++;
    strncpy(envValue, line+s, 1023);
    envValue[1023]='\0';
    ncclOsSetEnv(envVar, envValue);
    //printf("%s : %s->%s\n", fileName, envVar, envValue);
  }
  if (line) free(line);
  fclose(file);
}

// @EUGO_CHANGE: @begin:
// Changes:
// 1. Fixed issue w/ unconditionally set default conf file path
// 2. Using the different default conf file path `/etc/nccl.conf` -> `/usr/local/etc/nccl.conf`
//
// See original version below!
static void initEnvFunc() {
  // In the original implementation, first `NCCL_CONF_FILE` env var is checked and fills the NCCL configuration if it exists.
  // If it doesn't exist, it checks for the user home directory config file and fills the config if it exists.
  // Finally, it unconditionally merges the values from the default conf file path `/etc/nccl.conf` which is not ideal as it can override the previously set config values from the user home directory or `NCCL_CONF_FILE` env var.
  //
  // We've restructured the logic in a way that the first source found wins and multiple sources aren't merged together.
  // We've made it in a way that by default NCCL will ue our default config file but end-users will be able to override it via `NCCL_CONF_PATH` (and individual environment variables, if they work at all).
  // 1. `NCCL_CONF_FILE` env var
  // 2. `~/.nccl.conf`
  // 3. `/usr/local/etc/nccl.conf`
  //
  // File is only used if it exists in contrast to original logic which attempts to use the file even if it doesn't exist, overriding the previously set config to some extent.

  // 1. `NCCL_CONF_FILE` env var
  const char* userFile = std::getenv("NCCL_CONF_FILE");
  if (userFile && strlen(userFile) > 0) {
    const std::filesystem::path userFilePath{std::string(userFile)};
    // If the file is specified in `NCCL_CONF_FILE` env var but the pointed file doesn't exist, we fall through to the next configuration source.
    INFO(NCCL_ENV,"'NCCL_CONF_FILE' is set by environment to '%s' but doesn't exist. Skipping to the next source.", userFile);
    if (std::filesystem::exists(userFilePath)) {
      setEnvFile(strdup(userFilePath.string().c_str()));
      return;
    }
  }

  // 2. `~/.nccl.conf`
  const std::filesystem::path userDirPath{std::string(userHomeDir())};
  const std::filesystem::path userConfFilePath = userDirPath / ".nccl.conf";
  if (std::filesystem::exists(userConfFilePath)) {
    setEnvFile(strdup(userConfFilePath.string().c_str()));
    return;
  }

  // 3. `/usr/local/etc/nccl.conf`
  const std::filesystem::path defaultConfFilePath{"/usr/local/etc/nccl.conf"};
  if (std::filesystem::exists(defaultConfFilePath)) {
    setEnvFile(strdup(defaultConfFilePath.string().c_str()));
    return; // In case if more cases will be added in future.
  }
}

// @EUGO_ORIGINAL:
// static void initEnvFunc() {
//   char confFilePath[1024];
//   const char* userFile = std::getenv("NCCL_CONF_FILE");
//   if (userFile && strlen(userFile) > 0) {
//     snprintf(confFilePath, sizeof(confFilePath), "%s", userFile);
//     setEnvFile(confFilePath);
//   } else {
//     const char* userDir = userHomeDir();
//     if (userDir) {
//       snprintf(confFilePath, sizeof(confFilePath), "%s/.nccl.conf", userDir);
//       setEnvFile(confFilePath);
//     }
//   }
//   snprintf(confFilePath, sizeof(confFilePath), "/etc/nccl.conf");
//   setEnvFile(confFilePath);
// }
//
// @EUGO_CHANGE: @end


void initEnv() {
  static std::once_flag once;
  std::call_once(once, initEnvFunc);
}

void ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  if (COMPILER_ATOMIC_LOAD(cache, std::memory_order_relaxed) == uninitialized) {
    const char* str = ncclGetEnv(env);
    int64_t value = deftVal;
    if (str && strlen(str) > 0) {
      errno = 0;
      value = strtoll(str, nullptr, 0);
      if (errno) {
        value = deftVal;
        INFO(NCCL_ALL,"Invalid value %s for %s, using default %lld.", str, env, (long long)deftVal);
      } else {
        INFO(NCCL_ENV,"%s set by environment to %lld.", env, (long long)value);
      }
    }
    COMPILER_ATOMIC_STORE(cache, value, std::memory_order_relaxed);
  }
}

const char* ncclGetEnv(const char* name) {
  ncclInitEnv();
  return ncclEnvPluginGetEnv(name);
}
