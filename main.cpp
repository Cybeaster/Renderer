// #include "Application/Application.hpp"

// int main(int argc, char **argv)
// {
//     Application* app = Application::GetApplication();
// 	app->Start(argc,argv);
//     return 0;
// }
#include <iostream>
#include "ThreadPool.hpp"
using namespace RenderAPI::Functor;

int sum(int &ans, TTVector<int> &arr)
{
    for (int i = 0; i < arr.size(); ++i)
    {
        ans += arr[i];
    }
    return 0;
}

void OneParamFunc(int ans)
{
       std::cout<<ans<<std::endl;
}

void NoParamFunc()
{
    std::cout<<"Smthing"<<std::endl;
}

int main(int argc, char **argv)
{
    DECLARE_FUNCTOR(NoParamFunctor, void);
    DECLARE_FUNCTOR_OneParam(FunctorOneParam, void, int);

    RenderAPI::Thread::TThreadPool pool(3);

    TTVector<int> s1 = {1, 2, 3};
    int ans1 = 0;

    TTVector<int> s2 = {4, 5};
    int ans2 = 0;

    TTVector<int> s3 = {8, 9, 10};
    int ans3 = 0;

    auto funcNoParam = NoParamFunctor(NoParamFunc);
    auto funcOneParam = FunctorOneParam(OneParamFunc,ans1);
    funcNoParam();
    funcOneParam();
    auto id1 = pool.AddTask(std::move(funcNoParam));
    return 0;
}